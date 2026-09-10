MODULE module_ra_cloud_ml

! ========================================================================
! module_ra_cloud_ml.F
!
! Purpose: ML-based cloud-radiation residual corrections with energy closure
!          Phase 1: Empty hook that returns zero corrections
!          Phase 1b: Optional training data dump
!          Phase 2: MLP residual application (opt=1)
!          Phase 3: Energy closure via flux scaling
!
! MLP Architecture:
!   Input: 6 features (qc_col, qi_col, cldmax, var_col, coszen, albedo)
!   Hidden: 2 layers × 32 neurons (ReLU activation)
!   Output: 5 targets (dH_mean, dSWDNB, dSWUPB, dSWDNT, dSWUPT)
!   Preprocessing: StandardScaler (mean, scale)
!
! Energy Closure:
!   - Heating applied uniformly: rthratensw += dH_mean (preserves mass-weighted mean)
!   - Flux increments scaled to match heating: α = dH_mean / dH_flux
!   - dH_flux = ΔF_net / (cp * M), where M = sum(rho*dz), ΔF = (SWDNT-SWUPT) - (SWDNB-SWUPB)
!   - Scaling clamped to [0.25, 4] to avoid extreme rescaling
!   - Ensures column energy consistency between heating rate and net flux change
!
! Weight file: cloud_ml_sw_mlp.bin (float64 little-endian, C/row-major order)
!   Order: mean[6], scale[6], W0[6,32], b0[32], W1[32,32], b1[32], W2[32,5], b2[5]
!   Total: 1457 float64 values
!
! Forward pass: h0=(x-mean)/scale; h1=relu(h0@W0+b0); h2=relu(h1@W1+b1); y=h2@W2+b2
!
! Author: Cloud Agent
! Date: 2026-09-10
! ========================================================================

IMPLICIT NONE

PRIVATE
PUBLIC :: cloud_ml_sw_residual

! MLP architecture parameters
INTEGER, PARAMETER :: N_FEATURES = 6
INTEGER, PARAMETER :: N_HIDDEN1 = 32
INTEGER, PARAMETER :: N_HIDDEN2 = 32
INTEGER, PARAMETER :: N_OUTPUTS = 5
INTEGER, PARAMETER :: N_WEIGHTS = 1457  ! Total float64 values

! MLP weights (saved between calls)
REAL(8), DIMENSION(N_FEATURES), SAVE :: mlp_mean, mlp_scale
REAL(8), DIMENSION(N_FEATURES,N_HIDDEN1), SAVE :: mlp_W0
REAL(8), DIMENSION(N_HIDDEN1), SAVE :: mlp_b0
REAL(8), DIMENSION(N_HIDDEN1,N_HIDDEN2), SAVE :: mlp_W1
REAL(8), DIMENSION(N_HIDDEN2), SAVE :: mlp_b1
REAL(8), DIMENSION(N_HIDDEN2,N_OUTPUTS), SAVE :: mlp_W2
REAL(8), DIMENSION(N_OUTPUTS), SAVE :: mlp_b2

LOGICAL, SAVE :: weights_loaded = .FALSE.
LOGICAL, SAVE :: load_failed = .FALSE.

CONTAINS

! ========================================================================
! SUBROUTINE cloud_ml_sw_residual
!
! Purpose: Apply ML-based shortwave residual correction after RRTMG
!          opt=0: no-op
!          opt=1: apply MLP residual if weights present, else warn and no-op
!          Optionally dump training data when rad_cloud_ml_dump=1
!
! Arguments:
!   rad_cloud_ml_opt  - option flag (0=off, 1=apply MLP residual)
!   rad_cloud_ml_dump - dump flag (0=off, 1=dump training pairs)
!   itimestep         - current timestep number
!   id                - domain id
!   ids,ide,jds,jde,kds,kde - domain dimensions
!   ims,ime,jms,jme,kms,kme - memory dimensions  
!   its,ite,jts,jte,kts,kte - tile dimensions
!   
!   State inputs (3D: ims:ime,kms:kme,jms:jme):
!     t3d       - temperature (K)
!     p3d       - pressure (Pa)
!     rho3d     - density (kg/m3)
!     dz8w      - layer thickness (m)
!     qv3d      - water vapor mixing ratio (kg/kg)
!     qc3d      - cloud water mixing ratio (kg/kg)
!     qi3d      - cloud ice mixing ratio (kg/kg)
!     qs3d      - snow mixing ratio (kg/kg)
!     cldfra3d  - cloud fraction (0-1)
!     
!   State inputs (2D: ims:ime,jms:jme):
!     coszr     - cosine of solar zenith angle
!     albedo    - surface albedo (0-1)
!     
!   Inout fields to be modified (3D: ims:ime,kms:kme,jms:jme):
!     rthratensw - SW heating rate (K/s)
!     
!   Inout fields to be modified (2D: ims:ime,jms:jme):
!     swdnb     - SW down at bottom (W/m2)
!     swupb     - SW up at bottom (W/m2)
!     swdnt     - SW down at top (W/m2)
!     swupt     - SW up at top (W/m2)
!     gsw       - net SW at surface (W/m2)
!
!   Optional clear-sky fields:
!     rthratenswc - clear-sky SW heating rate (K/s)
!     swdnbc, swupbc, swdntc, swuptc - clear-sky fluxes
!
!   Optional microphysics effective radii (3D):
!     re_cloud, re_ice, re_snow - effective radii (microns)
!
! ========================================================================
SUBROUTINE cloud_ml_sw_residual(                                         &
       rad_cloud_ml_opt, rad_cloud_ml_dump,                              &
       itimestep, id,                                                    &
       ids,ide, jds,jde, kds,kde,                                        &
       ims,ime, jms,jme, kms,kme,                                        &
       its,ite, jts,jte, kts,kte,                                        &
       t3d, p3d, rho3d, dz8w,                                            &
       qv3d, qc3d, qi3d, qs3d, cldfra3d,                                 &
       coszr, albedo,                                                    &
       rthratensw,                                                       &
       swdnb, swupb, swdnt, swupt, gsw,                                  &
       rthratenswc, swdnbc, swupbc, swdntc, swuptc,                      &
       re_cloud, re_ice, re_snow                                         &
       )

   IMPLICIT NONE

   ! Arguments
   INTEGER, INTENT(IN) :: rad_cloud_ml_opt, rad_cloud_ml_dump
   INTEGER, INTENT(IN) :: itimestep, id
   INTEGER, INTENT(IN) :: ids,ide, jds,jde, kds,kde
   INTEGER, INTENT(IN) :: ims,ime, jms,jme, kms,kme
   INTEGER, INTENT(IN) :: its,ite, jts,jte, kts,kte
   
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: &
        t3d, p3d, rho3d, dz8w, qv3d, qc3d, qi3d, qs3d, cldfra3d
   
   REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: coszr, albedo
   
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: rthratensw
   
   REAL, DIMENSION(ims:ime,jms:jme), INTENT(INOUT) :: &
        swdnb, swupb, swdnt, swupt, gsw

   ! Optional clear-sky diagnostics
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), OPTIONAL, INTENT(IN) :: rthratenswc
   REAL, DIMENSION(ims:ime,jms:jme), OPTIONAL, INTENT(IN) :: &
        swdnbc, swupbc, swdntc, swuptc

   ! Optional microphysics effective radii
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), OPTIONAL, INTENT(IN) :: &
        re_cloud, re_ice, re_snow

   ! Local variables
   LOGICAL, SAVE :: first_call = .TRUE.
   LOGICAL, SAVE :: first_dump = .TRUE.

   ! Dump training data if requested (version 2: includes thermo/micro fields)
   IF (rad_cloud_ml_dump == 1) THEN
      CALL dump_training_data(                                           &
           itimestep, id,                                                &
           ids,ide, jds,jde, kds,kde,                                    &
           ims,ime, jms,jme, kms,kme,                                    &
           its,ite, jts,jte, kts,kte,                                    &
           qc3d, qi3d, qs3d, cldfra3d,                                   &
           coszr, albedo,                                                &
           rthratensw, swdnb, swupb, swdnt, swupt,                       &
           t3d, p3d, qv3d, rho3d, dz8w,                                  &
           re_cloud, re_ice, re_snow,                                    &
           first_dump                                                    &
           )
      first_dump = .FALSE.
   ENDIF

   ! Phase 2: MLP residual application
   IF (rad_cloud_ml_opt == 0) THEN
      ! Option disabled - immediate return
      RETURN
   ENDIF

   IF (rad_cloud_ml_opt == 1) THEN
      ! Load weights on first call
      IF (.NOT. weights_loaded .AND. .NOT. load_failed) THEN
         CALL load_mlp_weights()
         IF (first_call) THEN
            IF (weights_loaded) THEN
               WRITE(*,*) 'cloud_ml_sw_residual: opt=1 MLP weights loaded, applying residuals'
            ELSE
               WRITE(*,*) 'cloud_ml_sw_residual: opt=1 weight file missing, no corrections applied'
            ENDIF
            first_call = .FALSE.
         ENDIF
      ENDIF
      
      ! Apply MLP residuals if weights available
      IF (weights_loaded) THEN
         CALL apply_mlp_residuals(                                       &
              ids,ide, jds,jde, kds,kde,                                 &
              ims,ime, jms,jme, kms,kme,                                 &
              its,ite, jts,jte, kts,kte,                                 &
              qc3d, qi3d, qs3d, cldfra3d,                                &
              rho3d, dz8w,                                               &
              coszr, albedo,                                             &
              rthratensw, swdnb, swupb, swdnt, swupt, gsw                &
              )
      ENDIF
      RETURN
   ENDIF

   ! Invalid option value
   IF (first_call) THEN
      WRITE(*,*) 'WARNING: rad_cloud_ml_opt=', rad_cloud_ml_opt, ' not recognized'
      first_call = .FALSE.
   ENDIF
   
END SUBROUTINE cloud_ml_sw_residual

! ========================================================================
! SUBROUTINE dump_training_data
!
! Purpose: Dump radiation state for ML training (version 2: expanded fields)
!          Writes Fortran unformatted binary file with:
!          - Metadata: timestep, domain id, tile dimensions, format version
!          - 2D fields: coszen, albedo, SW fluxes
!          - 3D fields: qc, qi, qs, cldfra, rthratensw, t, p, qv, rho, dz, re_*
!
! File format: Fortran unformatted sequential, VERSION 2
!   Record 1: header (timestep, id, its,ite, jts,jte, kts,kte, version)
!   Record 2: coszr(its:ite,jts:jte)
!   Record 3: albedo(its:ite,jts:jte)
!   Record 4: qc3d(its:ite,kts:kte,jts:jte)
!   Record 5: qi3d(its:ite,kts:kte,jts:jte)
!   Record 6: qs3d(its:ite,kts:kte,jts:jte)
!   Record 7: cldfra3d(its:ite,kts:kte,jts:jte)
!   Record 8: rthratensw(its:ite,kts:kte,jts:jte)
!   Record 9: swdnb, swupb, swdnt, swupt (its:ite,jts:jte each)
!   === VERSION 2 ADDITIONS (Phase-A thermo/micro fields) ===
!   Record 10: t3d(its:ite,kts:kte,jts:jte)
!   Record 11: p3d(its:ite,kts:kte,jts:jte)
!   Record 12: qv3d(its:ite,kts:kte,jts:jte)
!   Record 13: rho3d(its:ite,kts:kte,jts:jte)
!   Record 14: dz8w(its:ite,kts:kte,jts:jte)
!   Record 15: re_cloud(its:ite,kts:kte,jts:jte) if present
!   Record 16: re_ice(its:ite,kts:kte,jts:jte) if present
!   Record 17: re_snow(its:ite,kts:kte,jts:jte) if present
!
! Python reader stub (version 2):
!   import numpy as np
!   with open('rad_ml_dump_*.bin', 'rb') as f:
!       header = np.fromfile(f, dtype=np.int32, count=9)
!       timestep, domid, its, ite, jts, jte, kts, kte, version = header
!       ni, nj, nk = ite-its+1, jte-jts+1, kte-kts+1
!       coszr = np.fromfile(f, dtype=np.float32, count=ni*nj).reshape(ni,nj)
!       albedo = np.fromfile(f, dtype=np.float32, count=ni*nj).reshape(ni,nj)
!       qc = np.fromfile(f, dtype=np.float32, count=ni*nk*nj).reshape(ni,nk,nj)
!       # ... qi, qs, cldfra, rthratensw, SW fluxes
!       # VERSION 2: t, p, qv, rho, dz, re_cloud, re_ice, re_snow
!
! ========================================================================
SUBROUTINE dump_training_data(                                           &
       itimestep, id,                                                    &
       ids,ide, jds,jde, kds,kde,                                        &
       ims,ime, jms,jme, kms,kme,                                        &
       its,ite, jts,jte, kts,kte,                                        &
       qc3d, qi3d, qs3d, cldfra3d,                                       &
       coszr, albedo,                                                    &
       rthratensw, swdnb, swupb, swdnt, swupt,                           &
       t3d, p3d, qv3d, rho3d, dz8w,                                      &
       re_cloud, re_ice, re_snow,                                        &
       first_dump                                                        &
       )

   IMPLICIT NONE

   ! Arguments
   INTEGER, INTENT(IN) :: itimestep, id
   INTEGER, INTENT(IN) :: ids,ide, jds,jde, kds,kde
   INTEGER, INTENT(IN) :: ims,ime, jms,jme, kms,kme
   INTEGER, INTENT(IN) :: its,ite, jts,jte, kts,kte
   
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: &
        qc3d, qi3d, qs3d, cldfra3d, rthratensw
   
   REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: &
        coszr, albedo, swdnb, swupb, swdnt, swupt

   ! Version 2: thermodynamic/microphysics fields
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: &
        t3d, p3d, qv3d, rho3d, dz8w
   
   ! Optional effective radii
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), OPTIONAL, INTENT(IN) :: &
        re_cloud, re_ice, re_snow

   LOGICAL, INTENT(IN) :: first_dump

   ! Local variables
   CHARACTER(LEN=256) :: filename
   INTEGER :: iunit, ierr
   INTEGER :: i, j, k
   INTEGER, PARAMETER :: DUMP_VERSION = 2
   LOGICAL :: has_cloudy_daylight
   REAL, PARAMETER :: COSZEN_MIN = 0.1  ! Minimum coszen for "daylight"
   REAL, PARAMETER :: CLOUD_MIN = 1.0E-6  ! Minimum total condensate for "cloudy"
   REAL :: total_cloud

   ! Check if we have cloudy + daylight conditions
   has_cloudy_daylight = .FALSE.
   DO j = jts, jte
      DO i = its, ite
         IF (coszr(i,j) > COSZEN_MIN) THEN
            total_cloud = 0.0
            DO k = kts, kte
               total_cloud = total_cloud + qc3d(i,k,j) + qi3d(i,k,j) + qs3d(i,k,j)
            END DO
            IF (total_cloud > CLOUD_MIN) THEN
               has_cloudy_daylight = .TRUE.
               EXIT
            ENDIF
         ENDIF
      END DO
      IF (has_cloudy_daylight) EXIT
   END DO

   ! Skip dump if no cloudy+daylight conditions
   IF (.NOT. has_cloudy_daylight) RETURN

   ! Log first dump
   IF (first_dump) THEN
      WRITE(*,*) 'cloud_ml_dump: Writing training data dumps to rad_ml_dump_*.bin (version 2)'
      WRITE(*,*) '  Format: Fortran unformatted sequential'
      WRITE(*,*) '  Fields: timestep, id, dims, coszen, albedo, qc/qi/qs, cldfra, rthratensw, SW fluxes'
      WRITE(*,*) '  + VERSION 2: t, p, qv, rho, dz, re_cloud, re_ice, re_snow'
   ENDIF

   ! Generate filename: rad_ml_dump_dDDD_tTTTTTTTT.bin
   WRITE(filename, '(A,I3.3,A,I8.8,A)') 'rad_ml_dump_d', id, '_t', itimestep, '.bin'

   ! Open file for unformatted sequential write
   iunit = 50 + id  ! Use domain-specific unit number
   OPEN(UNIT=iunit, FILE=TRIM(filename), FORM='UNFORMATTED', &
        ACCESS='SEQUENTIAL', STATUS='REPLACE', IOSTAT=ierr)
   
   IF (ierr /= 0) THEN
      WRITE(*,*) 'WARNING: cloud_ml_dump: Failed to open ', TRIM(filename)
      RETURN
   ENDIF

   ! Write header: version 2 includes format version field
   WRITE(iunit, IOSTAT=ierr) itimestep, id, its, ite, jts, jte, kts, kte, DUMP_VERSION
   IF (ierr /= 0) THEN
      WRITE(*,*) 'WARNING: cloud_ml_dump: Write error (header)'
      CLOSE(iunit)
      RETURN
   ENDIF

   ! Write 2D fields
   WRITE(iunit) ((coszr(i,j), i=its,ite), j=jts,jte)
   WRITE(iunit) ((albedo(i,j), i=its,ite), j=jts,jte)

   ! Write 3D fields (k is middle index following WRF convention)
   WRITE(iunit) (((qc3d(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   WRITE(iunit) (((qi3d(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   WRITE(iunit) (((qs3d(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   WRITE(iunit) (((cldfra3d(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   WRITE(iunit) (((rthratensw(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)

   ! Write SW flux fields
   WRITE(iunit) ((swdnb(i,j), i=its,ite), j=jts,jte)
   WRITE(iunit) ((swupb(i,j), i=its,ite), j=jts,jte)
   WRITE(iunit) ((swdnt(i,j), i=its,ite), j=jts,jte)
   WRITE(iunit) ((swupt(i,j), i=its,ite), j=jts,jte)

   ! === VERSION 2: thermodynamic/microphysics fields ===
   ! Write 3D thermo fields
   WRITE(iunit) (((t3d(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   WRITE(iunit) (((p3d(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   WRITE(iunit) (((qv3d(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   WRITE(iunit) (((rho3d(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   WRITE(iunit) (((dz8w(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)

   ! Write effective radii if present
   IF (PRESENT(re_cloud)) THEN
      WRITE(iunit) (((re_cloud(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   ENDIF
   IF (PRESENT(re_ice)) THEN
      WRITE(iunit) (((re_ice(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   ENDIF
   IF (PRESENT(re_snow)) THEN
      WRITE(iunit) (((re_snow(i,k,j), i=its,ite), k=kts,kte), j=jts,jte)
   ENDIF

   CLOSE(iunit)

   ! Log successful dump
   WRITE(*,*) 'cloud_ml_dump: Wrote ', TRIM(filename), ' (version 2)'

END SUBROUTINE dump_training_data

! ========================================================================
! SUBROUTINE load_mlp_weights
!
! Purpose: Load MLP weights from binary file
!          File: cloud_ml_sw_mlp.bin (float64 little-endian)
!          Order: mean[6], scale[6], W0[6,32], b0[32], W1[32,32], b1[32], W2[32,5], b2[5]
!
! Sets module variables: weights_loaded, load_failed
! ========================================================================
SUBROUTINE load_mlp_weights()

   IMPLICIT NONE

   CHARACTER(LEN=256), PARAMETER :: weight_file = 'cloud_ml_sw_mlp.bin'
   INTEGER :: iunit, ierr, ios
   REAL(8), DIMENSION(N_WEIGHTS) :: weight_buffer
   INTEGER :: idx, i, j

   ! Try to open weight file
   iunit = 99
   OPEN(UNIT=iunit, FILE=weight_file, FORM='UNFORMATTED', &
        ACCESS='STREAM', STATUS='OLD', IOSTAT=ios)
   
   IF (ios /= 0) THEN
      ! File not found
      load_failed = .TRUE.
      weights_loaded = .FALSE.
      RETURN
   ENDIF

   ! Read all weights as flat buffer
   READ(iunit, IOSTAT=ios) weight_buffer
   CLOSE(iunit)

   IF (ios /= 0) THEN
      WRITE(*,*) 'WARNING: cloud_ml: Failed to read weight file'
      load_failed = .TRUE.
      weights_loaded = .FALSE.
      RETURN
   ENDIF

   ! Unpack weights from buffer
   idx = 1
   
   ! mean[6]
   mlp_mean(:) = weight_buffer(idx:idx+N_FEATURES-1)
   idx = idx + N_FEATURES
   
   ! scale[6]
   mlp_scale(:) = weight_buffer(idx:idx+N_FEATURES-1)
   idx = idx + N_FEATURES
   
   ! W0[6,32] - row-major / C-order in buffer (matching Python W.ravel(order='C'))
   DO i = 1, N_FEATURES
      DO j = 1, N_HIDDEN1
         mlp_W0(i,j) = weight_buffer(idx)
         idx = idx + 1
      END DO
   END DO
   
   ! b0[32]
   mlp_b0(:) = weight_buffer(idx:idx+N_HIDDEN1-1)
   idx = idx + N_HIDDEN1
   
   ! W1[32,32] - row-major / C-order in buffer
   DO i = 1, N_HIDDEN1
      DO j = 1, N_HIDDEN2
         mlp_W1(i,j) = weight_buffer(idx)
         idx = idx + 1
      END DO
   END DO
   
   ! b1[32]
   mlp_b1(:) = weight_buffer(idx:idx+N_HIDDEN2-1)
   idx = idx + N_HIDDEN2
   
   ! W2[32,5] - row-major / C-order in buffer
   DO i = 1, N_HIDDEN2
      DO j = 1, N_OUTPUTS
         mlp_W2(i,j) = weight_buffer(idx)
         idx = idx + 1
      END DO
   END DO
   
   ! b2[5]
   mlp_b2(:) = weight_buffer(idx:idx+N_OUTPUTS-1)

   weights_loaded = .TRUE.
   load_failed = .FALSE.

END SUBROUTINE load_mlp_weights

! ========================================================================
! SUBROUTINE mlp_forward
!
! Purpose: MLP forward pass (2×32 hidden layers, ReLU)
!          Forward: h0=(x-mean)/scale; h1=relu(h0@W0+b0); h2=relu(h1@W1+b1); y=h2@W2+b2
!
! Input: features[N_FEATURES]
! Output: outputs[N_OUTPUTS]
! ========================================================================
SUBROUTINE mlp_forward(features, outputs)

   IMPLICIT NONE

   REAL(8), DIMENSION(N_FEATURES), INTENT(IN) :: features
   REAL(8), DIMENSION(N_OUTPUTS), INTENT(OUT) :: outputs

   REAL(8), DIMENSION(N_FEATURES) :: h0
   REAL(8), DIMENSION(N_HIDDEN1) :: h1
   REAL(8), DIMENSION(N_HIDDEN2) :: h2
   INTEGER :: i, j

   ! Input normalization: h0 = (x - mean) / scale
   DO i = 1, N_FEATURES
      h0(i) = (features(i) - mlp_mean(i)) / mlp_scale(i)
   END DO

   ! First hidden layer: h1 = relu(h0 @ W0 + b0)
   DO j = 1, N_HIDDEN1
      h1(j) = mlp_b0(j)
      DO i = 1, N_FEATURES
         h1(j) = h1(j) + h0(i) * mlp_W0(i,j)
      END DO
      h1(j) = MAX(0.0_8, h1(j))  ! ReLU
   END DO

   ! Second hidden layer: h2 = relu(h1 @ W1 + b1)
   DO j = 1, N_HIDDEN2
      h2(j) = mlp_b1(j)
      DO i = 1, N_HIDDEN1
         h2(j) = h2(j) + h1(i) * mlp_W1(i,j)
      END DO
      h2(j) = MAX(0.0_8, h2(j))  ! ReLU
   END DO

   ! Output layer: y = h2 @ W2 + b2
   DO j = 1, N_OUTPUTS
      outputs(j) = mlp_b2(j)
      DO i = 1, N_HIDDEN2
         outputs(j) = outputs(j) + h2(i) * mlp_W2(i,j)
      END DO
   END DO

END SUBROUTINE mlp_forward

! ========================================================================
! SUBROUTINE apply_mlp_residuals
!
! Purpose: Apply MLP residuals to cloudy+daylight columns with energy closure
!
! For each column with coszr > 0.1 and column condensate > 1e-6:
!   1. Extract 6 features: qc_col, qi_col, cldmax, var_col=0, coszen, albedo
!   2. Run MLP forward
!   3. Apply heating: dH_mean uniformly to all levels (preserves mass-weighted mean)
!   4. Apply flux closure: scale flux increments so net column energy change
!      matches applied heating (energy consistency)
!   5. Update gsw = swdnb - swupb
!
! Energy Closure (flux scaling):
!   - Compute net flux change: ΔF0 = (dSWDNT-dSWUPT) - (dSWDNB-dSWUPB)
!   - Compute column mass: M = sum_k(rho*dz)
!   - Implied heating from fluxes: dH_from_flux0 = ΔF0 / (cp * M)
!   - Scaling factor: α = dH_mean / dH_from_flux0 (clamped to [0.25, 4])
!   - Applied fluxes: sw* += α * dSW*
!   - This ensures column energy consistency between heating and flux increments
!
! Feature order (matching training export):
!   [0] qc_col = sum_k qc(i,k,j)
!   [1] qi_col = sum_k qi(i,k,j)
!   [2] cldmax = max_k cldfra(i,k,j)
!   [3] var_col = 0.0 (subgrid variance not available online)
!   [4] coszen = coszr(i,j)
!   [5] albedo = albedo(i,j)
!
! Output order:
!   [0] dH_mean   - uniform heating increment (K/s)
!   [1] dSWDNB    - bottom downwelling flux increment (W/m2)
!   [2] dSWUPB    - bottom upwelling flux increment (W/m2)
!   [3] dSWDNT    - top downwelling flux increment (W/m2)
!   [4] dSWUPT    - top upwelling flux increment (W/m2)
! ========================================================================
SUBROUTINE apply_mlp_residuals(                                          &
       ids,ide, jds,jde, kds,kde,                                        &
       ims,ime, jms,jme, kms,kme,                                        &
       its,ite, jts,jte, kts,kte,                                        &
       qc3d, qi3d, qs3d, cldfra3d,                                       &
       rho3d, dz8w,                                                      &
       coszr, albedo,                                                    &
       rthratensw, swdnb, swupb, swdnt, swupt, gsw                       &
       )

   IMPLICIT NONE

   INTEGER, INTENT(IN) :: ids,ide, jds,jde, kds,kde
   INTEGER, INTENT(IN) :: ims,ime, jms,jme, kms,kme
   INTEGER, INTENT(IN) :: its,ite, jts,jte, kts,kte
   
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: &
        qc3d, qi3d, qs3d, cldfra3d, rho3d, dz8w
   
   REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: coszr, albedo
   
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: rthratensw
   
   REAL, DIMENSION(ims:ime,jms:jme), INTENT(INOUT) :: &
        swdnb, swupb, swdnt, swupt, gsw

   ! Local variables
   REAL, PARAMETER :: COSZEN_MIN = 0.1
   REAL, PARAMETER :: CLOUD_MIN = 1.0E-6
   REAL, PARAMETER :: CP = 1004.5  ! J/(kg*K) - specific heat at constant pressure
   REAL, PARAMETER :: ALPHA_MIN = 0.25
   REAL, PARAMETER :: ALPHA_MAX = 4.0
   REAL, PARAMETER :: MASS_MIN = 1.0E-6  ! kg/m2 - minimum column mass
   REAL, PARAMETER :: TINY = 1.0E-12
   
   REAL(8), DIMENSION(N_FEATURES) :: features
   REAL(8), DIMENSION(N_OUTPUTS) :: mlp_outputs
   REAL :: qc_col, qi_col, qs_col, cldmax, total_cloud
   REAL :: column_mass, dF_net, dH_from_flux, alpha
   REAL :: dH_mean, dSWDNB, dSWUPB, dSWDNT, dSWUPT
   INTEGER :: i, j, k

   ! Loop over tile columns
   DO j = jts, jte
      DO i = its, ite
         
         ! Check daylight condition
         IF (coszr(i,j) <= COSZEN_MIN) CYCLE
         
         ! Extract column features
         qc_col = 0.0
         qi_col = 0.0
         qs_col = 0.0
         cldmax = 0.0
         
         DO k = kts, kte
            qc_col = qc_col + qc3d(i,k,j)
            qi_col = qi_col + qi3d(i,k,j)
            qs_col = qs_col + qs3d(i,k,j)
            cldmax = MAX(cldmax, cldfra3d(i,k,j))
         END DO
         
         total_cloud = qc_col + qi_col + qs_col
         
         ! Check cloudy condition
         IF (total_cloud <= CLOUD_MIN) CYCLE
         
         ! Build feature vector (matching training export order)
         features(1) = REAL(qc_col, 8)
         features(2) = REAL(qi_col, 8)
         features(3) = REAL(cldmax, 8)
         features(4) = 0.0_8  ! var_col placeholder (not available online)
         features(5) = REAL(coszr(i,j), 8)
         features(6) = REAL(albedo(i,j), 8)
         
         ! Run MLP forward
         CALL mlp_forward(features, mlp_outputs)
         
         ! Extract outputs
         dH_mean = REAL(mlp_outputs(1), 4)
         dSWDNB = REAL(mlp_outputs(2), 4)
         dSWUPB = REAL(mlp_outputs(3), 4)
         dSWDNT = REAL(mlp_outputs(4), 4)
         dSWUPT = REAL(mlp_outputs(5), 4)
         
         ! Apply heating: uniform dH_mean to all levels (preserves mass-weighted mean)
         DO k = kts, kte
            rthratensw(i,k,j) = rthratensw(i,k,j) + dH_mean
         END DO
         
         ! Energy closure: scale flux increments for consistency
         ! Compute column mass M = sum(rho * dz)
         column_mass = 0.0
         DO k = kts, kte
            column_mass = column_mass + rho3d(i,k,j) * dz8w(i,k,j)
         END DO
         
         IF (column_mass > MASS_MIN) THEN
            ! Net flux change from MLP outputs (W/m2)
            dF_net = (dSWDNT - dSWUPT) - (dSWDNB - dSWUPB)
            
            ! Implied heating from flux change (K/s)
            dH_from_flux = dF_net / (CP * column_mass)
            
            ! Compute scaling factor to match applied heating
            IF (ABS(dH_from_flux) > TINY) THEN
               alpha = dH_mean / dH_from_flux
               ! Clamp to reasonable range to avoid extreme rescaling
               alpha = MAX(ALPHA_MIN, MIN(ALPHA_MAX, alpha))
            ELSE
               ! If flux change is negligible, no scaling needed
               alpha = 1.0
            ENDIF
         ELSE
            ! If column mass too small, no scaling
            alpha = 1.0
         ENDIF
         
         ! Apply scaled flux increments
         swdnb(i,j) = swdnb(i,j) + alpha * dSWDNB
         swupb(i,j) = swupb(i,j) + alpha * dSWUPB
         swdnt(i,j) = swdnt(i,j) + alpha * dSWDNT
         swupt(i,j) = swupt(i,j) + alpha * dSWUPT
         
         ! Update GSW
         gsw(i,j) = swdnb(i,j) - swupb(i,j)
         
      END DO
   END DO

END SUBROUTINE apply_mlp_residuals

END MODULE module_ra_cloud_ml
