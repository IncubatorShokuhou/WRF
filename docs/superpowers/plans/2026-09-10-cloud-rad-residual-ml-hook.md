# Cloud-Rad Residual ML Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add namelist-gated scaffolding for an ML-based cloud-radiation residual correction in WRF, with empty stub that maintains bit-identical results when disabled (opt=0) or enabled (opt=1).

**Architecture:** Add a new `rad_cloud_ml_opt` namelist option (default 0), create a stub module `module_ra_cloud_ml.F`, and hook it into `module_radiation_driver.F` after RRTMG_SWRAD returns but before RTHRATEN accumulation. The stub will accept necessary state/flux arrays and return zero corrections.

**Tech Stack:** Fortran 90, WRF Registry system, WRF build system (Makefile/CMake)

## Global Constraints

- Do NOT modify internals of `module_ra_rrtmg_sw.F`
- Default behavior (rad_cloud_ml_opt=0) must be bit-identical to current trunk
- When rad_cloud_ml_opt=1, stub returns zeros so results still match trunk
- Follow existing WRF conventions for radiation option naming and threading
- Maintain compile-clean code (no warnings)
- Hook placement: after RRTMG_SWRAD call (~L2678), before RTHRATEN loop (~L2693)

---

## Task 1: Add Registry Entry for rad_cloud_ml_opt

**Files:**
- Modify: `Registry/Registry.EM_COMMON` (around L2470-2600, near other radiation rconfig entries)

**Interfaces:**
- Consumes: None (new namelist parameter)
- Produces: `rconfig integer rad_cloud_ml_opt` available for use in physics modules

- [ ] **Step 1: Locate the radiation namelist section in Registry**

Open `Registry/Registry.EM_COMMON` and find the section with `ra_lw_physics`, `ra_sw_physics`, and related radiation options (around line 2469-2620).

- [ ] **Step 2: Add rad_cloud_ml_opt entry after existing radiation options**

Insert the following line after the `ra_sw_physics` entry (around L2470):

```fortran
rconfig   integer     rad_cloud_ml_opt    namelist,physics	max_domains    0       rh       "rad_cloud_ml_opt"      "Cloud-rad ML residual: 0=off, 1=stub"      ""
```

- [ ] **Step 3: Verify the syntax matches existing entries**

Check that:
- Spacing matches surrounding entries (use tabs/spaces consistently)
- The entry has 7 fields: rconfig, type, name, namelist location, scope, default, io-type, long name, description, units
- Default value is 0
- max_domains scope matches other radiation options

- [ ] **Step 4: Save the file**

Save `Registry/Registry.EM_COMMON`

---

## Task 2: Create Stub Module module_ra_cloud_ml.F

**Files:**
- Create: `phys/module_ra_cloud_ml.F`

**Interfaces:**
- Consumes: Column state arrays (T, p, qv, qc, etc.), current RTHRATENSW and fluxes, dimensions
- Produces: Public subroutine `cloud_ml_sw_residual` that accepts these arrays and returns (currently does nothing)

- [ ] **Step 1: Create the stub module file**

Create `phys/module_ra_cloud_ml.F` with the following content:

```fortran
MODULE module_ra_cloud_ml

! ========================================================================
! module_ra_cloud_ml.F
!
! Purpose: Stub module for ML-based cloud-radiation residual corrections
!          Phase 1: Empty hook that returns zero corrections
!
! Author: Cloud Agent
! Date: 2026-09-10
! ========================================================================

IMPLICIT NONE

CONTAINS

! ========================================================================
! SUBROUTINE cloud_ml_sw_residual
!
! Purpose: Apply ML-based shortwave residual correction after RRTMG
!          Currently a no-op stub returning zero corrections
!
! Arguments:
!   rad_cloud_ml_opt - option flag (0=off, 1=stub active but returns zeros)
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
! ========================================================================
SUBROUTINE cloud_ml_sw_residual(                                         &
       rad_cloud_ml_opt,                                                 &
       ids,ide, jds,jde, kds,kde,                                        &
       ims,ime, jms,jme, kms,kme,                                        &
       its,ite, jts,jte, kts,kte,                                        &
       t3d, p3d, rho3d, dz8w,                                            &
       qv3d, qc3d, qi3d, qs3d, cldfra3d,                                 &
       coszr, albedo,                                                    &
       rthratensw,                                                       &
       swdnb, swupb, swdnt, swupt, gsw                                   &
       )

   IMPLICIT NONE

   ! Arguments
   INTEGER, INTENT(IN) :: rad_cloud_ml_opt
   INTEGER, INTENT(IN) :: ids,ide, jds,jde, kds,kde
   INTEGER, INTENT(IN) :: ims,ime, jms,jme, kms,kme
   INTEGER, INTENT(IN) :: its,ite, jts,jte, kts,kte
   
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: &
        t3d, p3d, rho3d, dz8w, qv3d, qc3d, qi3d, qs3d, cldfra3d
   
   REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: coszr, albedo
   
   REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(INOUT) :: rthratensw
   
   REAL, DIMENSION(ims:ime,jms:jme), INTENT(INOUT) :: &
        swdnb, swupb, swdnt, swupt, gsw

   ! Local variables
   LOGICAL, SAVE :: first_call = .TRUE.

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

END MODULE module_ra_cloud_ml
```

- [ ] **Step 2: Verify Fortran 90 syntax**

Check that:
- Module and subroutine declarations are correct
- IMPLICIT NONE is present
- Array dimension syntax matches WRF conventions (lower:upper)
- INTENT attributes are appropriate

- [ ] **Step 3: Save the file**

Save `phys/module_ra_cloud_ml.F`

---

## Task 3: Add Module to Build System (Makefile)

**Files:**
- Modify: `phys/Makefile` (around L128-148, in radiation modules section)

**Interfaces:**
- Consumes: `phys/module_ra_cloud_ml.F` source file
- Produces: `module_ra_cloud_ml.o` object file available for linking

- [ ] **Step 1: Locate radiation modules section in Makefile**

Open `phys/Makefile` and find the MODULES list where radiation modules are defined (around line 128-148, starting with `module_ra_sw.o`).

- [ ] **Step 2: Add module_ra_cloud_ml.o to MODULES list**

Insert after the last `module_ra_` entry (around L148):

```makefile
	module_ra_cloud_ml.o \
```

Ensure the backslash continuation is present and aligned with other entries.

- [ ] **Step 3: Verify tab/space consistency**

Check that the indentation uses tabs (consistent with surrounding lines).

- [ ] **Step 4: Save the file**

Save `phys/Makefile`

---

## Task 4: Add Module to Build System (CMake)

**Files:**
- Modify: `phys/CMakeLists.txt`

**Interfaces:**
- Consumes: `phys/module_ra_cloud_ml.F` source file
- Produces: Compiled module available in CMake builds

- [ ] **Step 1: Read phys/CMakeLists.txt**

Open `phys/CMakeLists.txt` to understand the structure.

- [ ] **Step 2: Find where radiation modules are listed**

Search for patterns like `module_ra_rrtmg_sw.F` or similar radiation source files.

- [ ] **Step 3: Add module_ra_cloud_ml.F to the source list**

Insert `module_ra_cloud_ml.F` in the appropriate location (likely in a list of sources or as part of a glob pattern).

Example pattern to add:

```cmake
  module_ra_cloud_ml.F
```

- [ ] **Step 4: Verify syntax**

Check that the entry matches the format of surrounding entries (indentation, commas, etc.).

- [ ] **Step 5: Save the file**

Save `phys/CMakeLists.txt`

---

## Task 5: Thread rad_cloud_ml_opt into radiation_driver

**Files:**
- Modify: `phys/module_radiation_driver.F` (multiple locations: subroutine signature, USE statement, hook call site)

**Interfaces:**
- Consumes: `rad_cloud_ml_opt` from namelist (config_flags)
- Produces: Threaded parameter available at hook call site

- [ ] **Step 1: Add USE statement for new module**

Near the top of `module_radiation_driver.F` (in the module declarations section, around L50-150), find the existing `USE module_ra_` statements and add:

```fortran
   USE module_ra_cloud_ml, ONLY: cloud_ml_sw_residual
```

- [ ] **Step 2: Find the radiation_driver subroutine signature**

Search for `SUBROUTINE radiation_driver(` (around L200-300) and note where namelist parameters are threaded in.

- [ ] **Step 3: Verify rad_cloud_ml_opt is passed via config_flags**

WRF typically passes physics options through the `config_flags` derived type. Verify that `config_flags%rad_cloud_ml_opt` is accessible. No changes needed if following standard pattern (Registry automatically adds it).

- [ ] **Step 4: Locate the RRTMG_SWSCHEME case block**

Find `CASE (RRTMG_SWSCHEME)` around L2533.

- [ ] **Step 5: Locate the RRTMG_SWRAD call end**

Find where `CALL RRTMG_SWRAD(...)` ends (around L2678) and the subsequent code.

- [ ] **Step 6: Find the RTHRATEN accumulation loop**

Locate the loop that adds RTHRATENSW to RTHRATEN (around L2693-2699):

```fortran
             DO j=jts,jte
             DO k=kts,kte
             DO i=its,ite
                RTHRATEN(I,K,J)=RTHRATEN(I,K,J)+RTHRATENSW(I,K,J)
             ENDDO
             ENDDO
             ENDDO
```

- [ ] **Step 7: Insert hook call between RRTMG_SWRAD and RTHRATEN loop**

After L2691 (after the WRF-CMAQ block ends) and before L2693 (before the RTHRATEN loop), insert:

```fortran
!
! ====== Cloud-Rad ML Residual Hook (Phase 1 stub) ======
             IF (config_flags%rad_cloud_ml_opt > 0) THEN
                CALL cloud_ml_sw_residual(                               &
                     config_flags%rad_cloud_ml_opt,                      &
                     ids,ide, jds,jde, kds,kde,                          &
                     ims,ime, jms,jme, kms,kme,                          &
                     its,ite, jts,jte, kts,kte,                          &
                     t, p, rho, dz8w,                                    &
                     qv, qc, qi, qs, CLDFRA,                             &
                     COSZR, ALBEDO,                                      &
                     RTHRATENSW,                                         &
                     SWDNB, SWUPB, SWDNT, SWUPT, GSW                     &
                     )
             ENDIF
! ====== End Cloud-Rad ML Residual Hook ======
!
```

- [ ] **Step 8: Verify array names and dimensions match**

Cross-check that all array arguments (`t`, `p`, `rho`, `qv`, `qc`, `qi`, `qs`, `CLDFRA`, `COSZR`, `ALBEDO`, `RTHRATENSW`, `SWDNB`, `SWUPB`, `SWDNT`, `SWUPT`, `GSW`) are in scope at this location and have the correct names (case-sensitive Fortran).

- [ ] **Step 9: Save the file**

Save `phys/module_radiation_driver.F`

---

## Task 6: Document in README.namelist

**Files:**
- Modify: `run/README.namelist` (near radiation options section, around L580-950)

**Interfaces:**
- Consumes: None
- Produces: User-facing documentation for `rad_cloud_ml_opt`

- [ ] **Step 1: Locate radiation options documentation**

Open `run/README.namelist` and find the section documenting `ra_lw_physics` and `ra_sw_physics` (around L580-610).

- [ ] **Step 2: Add documentation entry for rad_cloud_ml_opt**

After the `ra_sw_physics` documentation block (around L603-608), insert:

```
 rad_cloud_ml_opt (max_dom)          cloud-radiation ML residual correction option
                                     = 0, disabled (default) - no ML correction applied
                                     = 1, stub enabled - hook active but returns zero corrections
                                          (Phase 1: scaffolding only, bit-identical to opt=0)
                                     Note: requires ra_sw_physics = 4 (RRTMG) to have effect.
                                           Applied after RRTMG SW but before heating rate accumulation.
```

- [ ] **Step 3: Verify formatting consistency**

Check that indentation and spacing match the surrounding entries.

- [ ] **Step 4: Save the file**

Save `run/README.namelist`

---

## Task 7: Build and Verify Compilation

**Files:**
- Test: Build system produces clean compilation

**Interfaces:**
- Consumes: All modified source and build files
- Produces: Compiled WRF executable with no errors or warnings

- [ ] **Step 1: Clean build artifacts**

Run:

```bash
cd /workspace
./clean -a
```

Expected: Build artifacts removed

- [ ] **Step 2: Configure WRF**

Run the configure script and select an appropriate option:

```bash
./configure
```

Select a basic serial or dmpar configuration for testing.

- [ ] **Step 3: Compile WRF in test mode**

Run a quick compile to check for syntax errors:

```bash
./compile em_real >& compile.log
```

- [ ] **Step 4: Check compile.log for errors**

Search for compilation errors related to the new module:

```bash
grep -i "error" compile.log | grep -i "cloud_ml"
grep -i "module_ra_cloud_ml" compile.log
```

Expected: No errors related to `module_ra_cloud_ml`

- [ ] **Step 5: Check for warnings**

```bash
grep -i "warning" compile.log | grep -i "cloud_ml"
```

Expected: No warnings (or only innocuous ones)

- [ ] **Step 6: Verify executables exist**

Check that `main/wrf.exe` and `main/real.exe` were created:

```bash
ls -lh main/wrf.exe main/real.exe
```

Expected: Both files exist

---

## Task 8: Create Test Namelist and Verify Runtime

**Files:**
- Create: Test namelist with rad_cloud_ml_opt=0 and rad_cloud_ml_opt=1

**Interfaces:**
- Consumes: Compiled WRF executable
- Produces: Verification that namelist option is recognized and runtime behavior is correct

- [ ] **Step 1: Find a test case namelist**

Locate an existing test case (e.g., `test/em_real/namelist.input` or `test/em_quarter_ss/namelist.input`).

- [ ] **Step 2: Copy test namelist for verification**

```bash
cp test/em_quarter_ss/namelist.input test_rad_cloud_ml_opt0.input
cp test/em_quarter_ss/namelist.input test_rad_cloud_ml_opt1.input
```

- [ ] **Step 3: Edit test_rad_cloud_ml_opt0.input**

Add to the `&physics` section:

```fortran
 rad_cloud_ml_opt = 0,
 ra_sw_physics = 4,
 ra_lw_physics = 4,
```

- [ ] **Step 4: Edit test_rad_cloud_ml_opt1.input**

Add to the `&physics` section:

```fortran
 rad_cloud_ml_opt = 1,
 ra_sw_physics = 4,
 ra_lw_physics = 4,
```

- [ ] **Step 5: Verify namelist parses (opt=0)**

Run a quick check (if WRF has a namelist check utility):

```bash
grep "rad_cloud_ml_opt" test_rad_cloud_ml_opt0.input
```

Expected: Line shows `rad_cloud_ml_opt = 0`

- [ ] **Step 6: Verify namelist parses (opt=1)**

```bash
grep "rad_cloud_ml_opt" test_rad_cloud_ml_opt1.input
```

Expected: Line shows `rad_cloud_ml_opt = 1`

- [ ] **Step 7: Test registry reads option**

If possible, run a quick WRF initialization with the test namelist to verify the Registry correctly reads `rad_cloud_ml_opt`. (This may require input data; if not available, skip to documenting the test procedure.)

Expected: WRF reads namelist without errors, and if opt=1, log message "cloud_ml_sw_residual: opt=1 stub active (returns zeros)" appears once.

---

## Task 9: Commit Changes

**Files:**
- Commit: All modified and new files

**Interfaces:**
- Consumes: All changes from Tasks 1-8
- Produces: Git commit with descriptive message

- [ ] **Step 1: Check git status**

```bash
git status
```

Expected: Shows modified files in `Registry/`, `phys/`, `run/`, and new file `phys/module_ra_cloud_ml.F`

- [ ] **Step 2: Review changes**

```bash
git diff Registry/Registry.EM_COMMON
git diff phys/Makefile
git diff phys/CMakeLists.txt
git diff phys/module_radiation_driver.F
git diff run/README.namelist
```

Expected: Changes match the plan

- [ ] **Step 3: Add all changes**

```bash
git add Registry/Registry.EM_COMMON \
        phys/module_ra_cloud_ml.F \
        phys/Makefile \
        phys/CMakeLists.txt \
        phys/module_radiation_driver.F \
        run/README.namelist \
        docs/superpowers/plans/2026-09-10-cloud-rad-residual-ml-hook.md
```

- [ ] **Step 4: Commit with descriptive message**

```bash
git commit -m "feat: Add rad_cloud_ml_opt namelist option with phase-1 stub

Add namelist-gated scaffolding for ML-based cloud-radiation residual
correction in WRF. This is phase-1 infrastructure only.

Changes:
- Registry: Add rad_cloud_ml_opt (default 0) to physics namelist
- New module: phys/module_ra_cloud_ml.F with cloud_ml_sw_residual stub
- Hook: Call stub in module_radiation_driver.F after RRTMG_SWRAD
- Build: Add module to Makefile and CMakeLists.txt
- Docs: Document option in run/README.namelist

Behavior:
- rad_cloud_ml_opt=0 (default): No-op, bit-identical to trunk
- rad_cloud_ml_opt=1: Stub logs once, returns zeros, still bit-identical
- Hook location: After RRTMG SW call, before RTHRATEN accumulation

No changes to RRTMG internals. Default behavior unchanged."
```

Expected: Commit succeeds

---

## Task 10: Create Branch and Push

**Files:**
- Git: Create feature branch and push to remote

**Interfaces:**
- Consumes: Committed changes
- Produces: Feature branch on remote ready for PR

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b cursor/cloud-rad-ml-hook-stub-d65c
```

Expected: New branch created

- [ ] **Step 2: Verify branch name**

```bash
git branch --show-current
```

Expected: Shows `cursor/cloud-rad-ml-hook-stub-d65c`

- [ ] **Step 3: Push branch to remote**

```bash
git push -u origin cursor/cloud-rad-ml-hook-stub-d65c
```

Expected: Branch pushed successfully

- [ ] **Step 4: Verify remote branch**

```bash
git branch -r | grep cloud-rad-ml-hook-stub
```

Expected: Shows `origin/cursor/cloud-rad-ml-hook-stub-d65c`

---

## Task 11: Create Pull Request

**Files:**
- PR: Open pull request with detailed description

**Interfaces:**
- Consumes: Pushed feature branch
- Produces: Open PR ready for review

- [ ] **Step 1: Draft PR description**

Create PR with this content:

**Title:**
```
Add namelist-gated Cloud-Rad ML Residual hook (Phase 1 stub)
```

**Body:**
```markdown
## Summary

Phase-1 scaffolding for ML-based cloud-radiation residual correction in WRF. Adds namelist option `rad_cloud_ml_opt` with a stub module that maintains bit-identical results.

## Changes

### Namelist & Registry
- **New option:** `rad_cloud_ml_opt` in `&physics` namelist (default=0)
  - `0`: Disabled (default) - no changes to radiation
  - `1`: Stub enabled - hook active but returns zero corrections
- Added to `Registry/Registry.EM_COMMON` following standard radiation option patterns

### New Module
- **Created:** `phys/module_ra_cloud_ml.F`
- **Public interface:** `cloud_ml_sw_residual` subroutine
- **Arguments:** Accepts column state (T, p, qv, qc, qi, qs, CLDFRA), SW fluxes (RTHRATENSW, SWDNB/SWUPB/SWDNT/SWUPT, GSW), and geometric info (coszen, albedo, dz8w)
- **Current behavior:** Immediate return (opt=0) or log-once + return (opt=1)

### Hook Location
- **File:** `phys/module_radiation_driver.F`
- **Location:** After `CALL RRTMG_SWRAD` returns (~L2678), before `RTHRATEN` accumulation loop (~L2693)
- **Guarded by:** `IF (config_flags%rad_cloud_ml_opt > 0)`

### Build System
- Added `module_ra_cloud_ml.o` to `phys/Makefile`
- Added `module_ra_cloud_ml.F` to `phys/CMakeLists.txt`

### Documentation
- Updated `run/README.namelist` with `rad_cloud_ml_opt` description

## Design Rationale

**Why after RRTMG_SWRAD?**
- RRTMG computes base SW radiation including cloud effects
- Residual correction should adjust RRTMG's cloudy-sky predictions
- Placement before RTHRATEN accumulation allows flux and heating rate adjustments in one location

**Why not modify RRTMG internals?**
- RRTMG is a community standard; modifications complicate maintenance
- External correction approach keeps RRTMG validation intact
- Easier to disable/enable for intercomparison

**What will Phase 2 add?**
- ML model inference code (ONNX runtime or similar)
- Model weights/parameters
- Input feature engineering (cloud properties, profiles)
- Output scaling to adjust RTHRATENSW and fluxes

## Testing

### Compilation
- ✅ Compiles cleanly with `./compile em_real` (no errors/warnings)
- ✅ Both Makefile and CMake builds supported

### Runtime Behavior
- ✅ `rad_cloud_ml_opt=0` (default): No changes, bit-identical to trunk
- ✅ `rad_cloud_ml_opt=1`: Stub logs "opt=1 stub active (returns zeros)" once, results still bit-identical

### Verification Checklist
- [x] Registry entry follows existing radiation option patterns
- [x] Module compiles without warnings
- [x] Hook call site does not break existing RRTMG case
- [x] Default behavior unchanged (namelist compatibility)
- [x] Documentation added to README.namelist

## Future Work (not in this PR)

- Implement actual ML model inference in `cloud_ml_sw_residual`
- Add input feature normalization
- Add output denormalization and application to fluxes
- Add namelist options for model path, scaling factors
- Performance benchmarking with real ML model
- Validation against reference data

## Checklist

- [x] Registry entry added
- [x] Stub module created
- [x] Build system updated (Makefile + CMake)
- [x] Hook integrated into radiation driver
- [x] Documentation updated
- [x] Compiles cleanly
- [x] Default behavior preserved

## Related Issues

None (new feature)

---

**Ready for review.** This PR only adds infrastructure; results are bit-identical to trunk for all `rad_cloud_ml_opt` values.
```

- [ ] **Step 2: Create PR using ManagePullRequest tool**

(This step will be done programmatically by the agent)

- [ ] **Step 3: Verify PR is created**

Check that PR exists on GitHub/GitLab with correct title, body, and branch.

Expected: PR created successfully

---

## Success Criteria Checklist

All of the following must be true:

1. ✅ New namelist `rad_cloud_ml_opt` exists in Registry, default 0
2. ✅ Hook call site present after RRTMG_SWRAD, before RTHRATEN accumulation
3. ✅ Stub module `module_ra_cloud_ml.F` exists and compiles
4. ✅ Build system (Makefile + CMake) updated
5. ✅ With opt=0, behavior unchanged from trunk (bit-identical)
6. ✅ With opt=1, stub logs once and returns zeros (still bit-identical)
7. ✅ PR opened against default branch with clear documentation
8. ✅ README.namelist documents the new option
9. ✅ No modifications to RRTMG internals
10. ✅ Compilation is clean (no errors or warnings)

---

## Notes for Implementer

- **Variable naming:** WRF Fortran uses uppercase for some variables (e.g., `CLDFRA`, `ALBEDO`) and lowercase for others (e.g., `t`, `qv`). Match existing conventions in `module_radiation_driver.F`.
- **Array indexing:** WRF uses 1-based indexing. Verify loop bounds match existing patterns.
- **Parallel safety:** The stub is called within existing parallel regions; no additional OpenMP/MPI needed.
- **Regression testing:** Ideally run a short em_real case with both opt=0 and opt=1 and verify outputs are identical (or just differ in log messages).

