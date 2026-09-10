MODULE module_ra_cloud_ml

! ========================================================================
! module_ra_cloud_ml.F
!
! Purpose: Stub module for ML-based cloud-radiation residual corrections
!          Phase 1: Empty hook that returns zero corrections
!          Phase 1b: Optional training data dump
!
! Author: Cloud Agent
! Date: 2026-09-10
! ========================================================================

IMPLICIT NONE

PRIVATE
PUBLIC :: cloud_ml_sw_residual

CONTAINS

! ========================================================================
! SUBROUTINE cloud_ml_sw_residual
!
! Purpose: Apply ML-based shortwave residual correction after RRTMG
!          Currently a no-op stub returning zero corrections
!          Optionally dump training data when rad_cloud_ml_dump=1
!
! Arguments:
!   rad_cloud_ml_opt  - option flag (0=off, 1=stub active but returns zeros)
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
       rthratenswc, swdnbc, swupbc, swdntc, swuptc                       &
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

   ! Local variables
   LOGICAL, SAVE :: first_call = .TRUE.
   LOGICAL, SAVE :: first_dump = .TRUE.

   ! Dump training data if requested
   IF (rad_cloud_ml_dump == 1) THEN
      CALL dump_training_data(                                           &
           itimestep, id,                                                &
           ids,ide, jds,jde, kds,kde,                                    &
           ims,ime, jms,jme, kms,kme,                                    &
           its,ite, jts,jte, kts,kte,                                    &
           qc3d, qi3d, qs3d, cldfra3d,                                   &
           coszr, albedo,                                                &
           rthratensw, swdnb, swupb, swdnt, swupt,                       &
           first_dump                                                    &
           )
      first_dump = .FALSE.
   ENDIF

   ! Phase 1 stub: do nothing, return immediately
   IF (rad_cloud_ml_opt == 0) THEN
      ! Option disabled - immediate return
      RETURN
   ENDIF

   IF (rad_cloud_ml_opt == 1) THEN
      ! Option enabled but stub only - log once and return
      IF (first_call) THEN
         WRITE(*,*) 'cloud_ml_sw_residual: opt=1 stub active (returns zeros)'
         first_call = .FALSE.
      ENDIF
      ! Return zero corrections (no-op)
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
! Purpose: Dump radiation state for ML training
!          Writes Fortran unformatted binary file with:
!          - Metadata: timestep, domain id, tile dimensions
!          - 2D fields: coszen, albedo, SW fluxes
!          - 3D fields: qc, qi, qs, cldfra, rthratensw
!
! File format: Fortran unformatted sequential
!   Record 1: header (timestep, id, its,ite, jts,jte, kts,kte)
!   Record 2: coszr(its:ite,jts:jte)
!   Record 3: albedo(its:ite,jts:jte)
!   Record 4: qc3d(its:ite,kts:kte,jts:jte)
!   Record 5: qi3d(its:ite,kts:kte,jts:jte)
!   Record 6: qs3d(its:ite,kts:kte,jts:jte)
!   Record 7: cldfra3d(its:ite,kts:kte,jts:jte)
!   Record 8: rthratensw(its:ite,kts:kte,jts:jte)
!   Record 9: swdnb, swupb, swdnt, swupt (its:ite,jts:jte each)
!
! Python reader stub:
!   import numpy as np
!   with open('rad_ml_dump_*.bin', 'rb') as f:
!       header = np.fromfile(f, dtype=np.int32, count=7)
!       timestep, domid, its, ite, jts, jte, kts, kte = header
!       ni, nj, nk = ite-its+1, jte-jts+1, kte-kts+1
!       coszr = np.fromfile(f, dtype=np.float32, count=ni*nj).reshape(ni,nj)
!       # ... similar for other fields
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

   LOGICAL, INTENT(IN) :: first_dump

   ! Local variables
   CHARACTER(LEN=256) :: filename
   INTEGER :: iunit, ierr
   INTEGER :: i, j, k
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
      WRITE(*,*) 'cloud_ml_dump: Writing training data dumps to rad_ml_dump_*.bin'
      WRITE(*,*) '  Format: Fortran unformatted sequential'
      WRITE(*,*) '  Fields: timestep, id, dims, coszen, albedo, qc/qi/qs, cldfra, rthratensw, SW fluxes'
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

   ! Write header
   WRITE(iunit, IOSTAT=ierr) itimestep, id, its, ite, jts, jte, kts, kte
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

   CLOSE(iunit)

   ! Log successful dump
   WRITE(*,*) 'cloud_ml_dump: Wrote ', TRIM(filename)

END SUBROUTINE dump_training_data

END MODULE module_ra_cloud_ml
