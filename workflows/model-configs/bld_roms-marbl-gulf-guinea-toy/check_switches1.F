#include "cppdefs.opt"


      subroutine check_switches1(ierr)

!!!!!! WARNING: THIS IS A MACHINE GENERATED CODE, DO NOT EDIT! !!!!!!
!!!!!! This file needs to be updated only if new CPP-switches  !!!!!!
!!!!!! were introduced into "cppdefs.opt".  NO ACTION IS NEEDED  !!!!!!
!!!!!! if changes in "cppdefs.opt" are limited to activation or  !!!!!!
!!!!!! deactivation of previously known switches.              !!!!!!
!!!!!! To refresh this file compile and execute "cppcheck.F"   !!!!!!
!!!!!! as an independent program, or use commands              !!!!!!
!!!!!! "make checkdefs" or "make depend".                      !!!!!!
!!!!!! Number of Configuration Choices:                      1 !!!!!!
!!!!!! Total number of CPP-switches:                        69 !!!!!!

      use param
      use scalars
      use strings
      implicit none
      integer ierr, is,ie, iexample
      do is=1,max_opt_size
        cpps(is:is)=' '
      enddo
      iexample=0
!
! cppdefs.opt                                                     
!
      is=1 ; ie=is+12
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='<cppdefs.opt>'
      cpps(ie+1:ie+1)=' '

#ifdef PACIFIC_PD
      iexample=iexample+1
      is=ie+2 ; ie=is+9
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='PACIFIC_PD'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef USWC_sample
      is=ie+2 ; ie=is+10
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='USWC_sample'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SOLVE3D
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SOLVE3D'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef UV_ADV
      is=ie+2 ; ie=is+5
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='UV_ADV'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef UV_COR
      is=ie+2 ; ie=is+5
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='UV_COR'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef ADV_ISONEUTRAL
      is=ie+2 ; ie=is+13
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='ADV_ISONEUTRAL'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef NONLIN_EOS
      is=ie+2 ; ie=is+9
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='NONLIN_EOS'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SPLIT_EOS
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SPLIT_EOS'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SALINITY
      is=ie+2 ; ie=is+7
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SALINITY'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef BULK_FRC
      is=ie+2 ; ie=is+7
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='BULK_FRC'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef DIURNAL_SRFLUX
      is=ie+2 ; ie=is+13
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='DIURNAL_SRFLUX'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef VFLX_CORR
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='VFLX_CORR'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef QCORRECTION
      is=ie+2 ; ie=is+10
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='QCORRECTION'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef T_FRC_BRY
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='T_FRC_BRY'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef Z_FRC_BRY
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='Z_FRC_BRY'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef M3_FRC_BRY
      is=ie+2 ; ie=is+9
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='M3_FRC_BRY'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef M2_FRC_BRY
      is=ie+2 ; ie=is+9
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='M2_FRC_BRY'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SPONGE
      is=ie+2 ; ie=is+5
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SPONGE'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SPONGE_WIDTH
      is=ie+2 ; ie=is+11
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SPONGE_WIDTH'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef UV_VIS2
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='UV_VIS2'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef TS_DIF2
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='TS_DIF2'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef LMD_MIXING
      is=ie+2 ; ie=is+9
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='LMD_MIXING'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef LMD_KPP
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='LMD_KPP'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef LMD_NONLOCAL
      is=ie+2 ; ie=is+11
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='LMD_NONLOCAL'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef LMD_RIMIX
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='LMD_RIMIX'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef LMD_CONVEC
      is=ie+2 ; ie=is+9
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='LMD_CONVEC'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef LMD_BKPP
      is=ie+2 ; ie=is+7
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='LMD_BKPP'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef CURVGRID
      is=ie+2 ; ie=is+7
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='CURVGRID'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SPHERICAL
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SPHERICAL'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef MASKING
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='MASKING'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef MASK_LAND_DATA
      is=ie+2 ; ie=is+13
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='MASK_LAND_DATA'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef OBC_M2FLATHER
      is=ie+2 ; ie=is+12
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='OBC_M2FLATHER'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef OBC_M3ORLANSKI
      is=ie+2 ; ie=is+13
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='OBC_M3ORLANSKI'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef OBC_TORLANSKI
      is=ie+2 ; ie=is+12
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='OBC_TORLANSKI'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef OBC_WEST
      is=ie+2 ; ie=is+7
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='OBC_WEST'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef OBC_NORTH
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='OBC_NORTH'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef OBC_EAST
      is=ie+2 ; ie=is+7
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='OBC_EAST'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef OBC_SOUTH
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='OBC_SOUTH'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef BIOLOGY_BEC2
      is=ie+2 ; ie=is+11
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='BIOLOGY_BEC2'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef BIOLOGY
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='BIOLOGY'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef DEFAULT_BRY_VALUES
      is=ie+2 ; ie=is+17
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='DEFAULT_BRY_VALUES'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef BEC2_DIAG
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='BEC2_DIAG'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef Ncycle_SY
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='Ncycle_SY'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef N2O_TRACER_DECOMP
      is=ie+2 ; ie=is+16
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='N2O_TRACER_DECOMP'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef N2O_NEV
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='N2O_NEV'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef RIVER_LOAD_N
      is=ie+2 ; ie=is+11
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='RIVER_LOAD_N'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef RIVER_LOAD_P
      is=ie+2 ; ie=is+11
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='RIVER_LOAD_P'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef PCO2AIR_FORCING
      is=ie+2 ; ie=is+14
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='PCO2AIR_FORCING'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef BEC_COCCO
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='BEC_COCCO'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef TDEP_REMIN
      is=ie+2 ; ie=is+9
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='TDEP_REMIN'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef DAILYPAR_BEC
      is=ie+2 ; ie=is+11
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='DAILYPAR_BEC'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef DAILYPAR_PHOTOINHIBITION
      is=ie+2 ; ie=is+23
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='DAILYPAR_PHOTOINHIBITION'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef TIDES
      is=ie+2 ; ie=is+4
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='TIDES'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef POT_TIDES
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='POT_TIDES'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SSH_TIDES
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SSH_TIDES'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef UV_TIDES
      is=ie+2 ; ie=is+7
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='UV_TIDES'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef WEC
      is=ie+2 ; ie=is+2
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='WEC'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef BRK0
      is=ie+2 ; ie=is+3
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='BRK0'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SURFACE_BREAK
      is=ie+2 ; ie=is+12
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SURFACE_BREAK'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SPEC_DD
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SPEC_DD'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef LOG_BDRAG
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='LOG_BDRAG'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef WKB_WWAVE
      is=ie+2 ; ie=is+8
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='WKB_WWAVE'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef BBL
      is=ie+2 ; ie=is+2
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='BBL'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef WAVE_OFFLINE
      is=ie+2 ; ie=is+11
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='WAVE_OFFLINE'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef WAVE_FRICTION
      is=ie+2 ; ie=is+12
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='WAVE_FRICTION'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef BBL_S95
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='BBL_S95'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef SUP_OFF
      is=ie+2 ; ie=is+6
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='SUP_OFF'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef WAVE_DIFF_FROM_LM
      is=ie+2 ; ie=is+16
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='WAVE_DIFF_FROM_LM'
      cpps(ie+1:ie+1)=' '
#endif
#ifdef DUMMY_CASE
      is=ie+2 ; ie=is+9
      if (ie >= max_opt_size) goto 99
      cpps(is:ie)='DUMMY_CASE'
      cpps(ie+1:ie+1)=' '
#endif
      if (iexample == 0) then
        mpi_master_only write(*,'(/1x,A)')
     & '### ERROR in "cppdefs.opt": no configuration is specified.'
        ierr=ierr+1
      elseif (iexample > 1) then
        mpi_master_only write(*,'(/1x,2A/)')
     &  '### ERROR: More than one main configuration ',
     &             'switch is defined in "cppdefs.opt".'
        ierr=ierr+1
      endif
      return
  99  mpi_master_only write(*,'(/1x,2A/14x,A)')
     &  '### ERROR: checkdefs :: Insufficient size of string "cpps" ',
     &  'in file "strings".', 'Increase the size it and recompile.'
      ierr=ierr+1
      return
      end
