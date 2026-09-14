from Run_ATES import run_DARTS
from Run_ATES import create_geomodel_confined_disc_opt

### Model dimensions
nx = 50 # correct would be 200
ny = 50 # correct would be 200
nz = 4

nx_large = 25 # correct would be 75
ny_large = 25 # correct would be 75

dx = 1
dy = 1
dz = 2

dx_large = 4
dy_large = 4

### Properties
permeability = 3.65 * 1e-11 # m2
permeability_mD = permeability * 1.01325 * 1e15 #1e12 m2 to D, 1e3
print("Horizontal permeability", permeability_mD, "mD")
anisotropy = 5
permeability_v = permeability_mD / anisotropy
print("Vertical permeability", permeability_v, "mD")

permeability_AT = 7.3 * 1e-14 # m2
permeability_mD_AT = permeability_AT * 1013250000000000
print("Horizontal permeability aquitard", permeability_mD_AT, "mD")
anisotropy_AT = 5
permeability_v_AT = permeability_mD_AT / anisotropy_AT
print("Vertical permeability aquitard", permeability_v_AT, "mD")

porosity = 0.3

solid_density = 2640 #kg/m3
water_density = 998 #kg/m3

solid_heat_cap = 710 #J/kg/K
water_heat_cap = 4184 #J/kg/K

volumetric_heat_cap = ((porosity * water_density * water_heat_cap) + ((1 - porosity) * solid_density * solid_heat_cap))/ 1000 #kJ/m3
print("Volumetric heat capacity", volumetric_heat_cap, "kJ/m3")

l_w = 0.57 #W/m/K
l_s = 2 #W/m/K
l_c = 1.7 #W/m/K
t_c_sand = porosity * l_w + (1 - porosity) * l_s #W/m/K
t_c_clay = porosity * l_w + (1 - porosity) * l_c #W/m/K
thermal_conductivity_DARTS = t_c_sand * (24 * 3600) / 1000 #kJ/K/day
print("Thermal conductivity", thermal_conductivity_DARTS, "kJ/K/day")
thermal_conductivity_DARTS_AT = t_c_clay * (24 * 3600) / 1000 #kJ/K/day
print("Thermal conductivity aquitard", thermal_conductivity_DARTS_AT, "kJ/K/day")

### Timestepping & Operational profile
# Block-function Operational profile
daysprofile = [5, 5, 5, 5] #[90, 90, 90, 90]  # (days)
storage_periods = ['Charge', 'Rest', 'Discharge', 'Rest']

#daysprofile_v2 = []
#storage_periods_v2 = []

#for i in range(len(daysprofile)):
    #length = daysprofile[i]
    #for j in range(length):
        #daysprofile_v2.append(1)
        #storage_periods_v2.append(storage_periods[i])

dt_max = 1 #day
dt_mult = 4

### Simulation settings
Tin = 20 # deg C
Q = 2222 # m3/day
well_diameter = 1 # m

var_mod = create_geomodel_confined_disc_opt(nx, ny, nx_large, ny_large,
                                            20, 11, 1,
                                            dx, dy, dz, dx_large, dy_large,
                                            0,
                                            permeability_mD, permeability_v, porosity, volumetric_heat_cap, thermal_conductivity_DARTS,
                                            permeability_mD_AT, permeability_v_AT, porosity, volumetric_heat_cap, thermal_conductivity_DARTS_AT,
                                            int((nx/2) - 1 + nx_large), int((ny/2) - 1 + ny_large))


run_DARTS ("case_synthetic_no_obswell_gridext",
           var_mod,
               1,
               20, 5,
               Q, daysprofile, storage_periods,
           set_transition_runtime = 1e-3,
               well_diameter = 1,
                n_points = 256) #ensure that n-points is correctly inferred in the other scripts