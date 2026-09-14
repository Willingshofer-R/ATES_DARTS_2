# Load model creation
from ATES_model import Model

import os
import xarray as xr
import pandas as pd
import numpy as np

def create_geomodel_confined_disc_opt(nx, ny, nx_large, ny_large,
                                      nz_aq, nz_conf, nz_actnum,
                                      dx, dy, dz, dx_large, dy_large,
                                      start_z,
                                      perm_h, perm_v, porosity, h_cap, t_cond,
                                      perm_h_conf, perm_v_conf, porosity_conf, h_cap_conf, t_cond_conf,
                                      hwx, hwy):

    "nz_conf includes the nz_actnum"

    #create dx arrays
    dx_l = np.tile(dx_large, nx_large)
    dx_c = np.tile(dx, nx)
    dx_r = np.tile(dx_large, nx_large)
    dx_full = np.concatenate([dx_l, dx_c, dx_r])

    x_cell_edges = np.insert(np.cumsum(dx_full), 0 ,0)

    x_cell_centers = []
    for i in range(len(x_cell_edges) - 1):
        x_edge_left = x_cell_edges[i]
        x_edge_right = x_cell_edges[i + 1]
        x_center_val = (x_edge_left + x_edge_right)/2
        x_cell_centers.append(x_center_val)

    # create dy arrays
    dy_d = np.tile(dy_large, ny_large)
    dy_c = np.tile(dy, ny)
    dy_u = np.tile(dy_large, ny_large)
    dy_full = np.concatenate([dy_d, dy_c, dy_u])

    y_cell_edges = np.insert(np.cumsum(dy_full), 0, 0)

    y_cell_centers = []
    for i in range(len(y_cell_edges) - 1):
        y_edge_left = y_cell_edges[i]
        y_edge_right = y_cell_edges[i + 1]
        y_center_val = (y_edge_left + y_edge_right) / 2
        y_cell_centers.append(y_center_val)

    # directional cell-count
    nx_full = len(x_cell_centers)
    ny_full = len(y_cell_centers)

    z = np.arange(nz_conf + nz_aq + nz_conf + 1) * dz + start_z #+ dz / 2
    z_center = z[:-1] + 0.5 * dz
    dz_full = np.tile(2, len(z_center))
    print("Domain z:", min(z), max(z))

    # Property arrays
    permeability_xy_cells = np.full((nx_full, ny_full, nz_aq), perm_h)
    permeability_z_cells  = np.full((nx_full, ny_full, nz_aq), perm_v)
    porosity_cells        = np.full((nx_full, ny_full, nz_aq), porosity)
    heat_capacity_cells   = np.full((nx_full, ny_full, nz_aq), h_cap)
    thermal_conductivity_cells = np.full((nx_full, ny_full, nz_aq), t_cond)

    if nz_actnum > nz_conf:
        print("nz_actnum should not be larger than nz_conf")

    act_num1 = np.full((nx_full, ny_full, nz_aq + 2 * (nz_conf - nz_actnum)), 1)

    # Confining layers
    conf_permeability_xy_cells = np.full((nx_full, ny_full, nz_conf), perm_h_conf)
    conf_permeability_z_cells  = np.full((nx_full, ny_full, nz_conf), perm_v_conf)
    conf_porosity_cells        = np.full((nx_full, ny_full, nz_conf), porosity_conf)
    conf_heat_capacity_cells   = np.full((nx_full, ny_full, nz_conf), h_cap_conf)
    conf_thermal_conductivity_cells = np.full((nx_full, ny_full, nz_conf), t_cond_conf)

    act_num0 = np.full((nx_full, ny_full, nz_actnum), 0)

    tot_permeability_xy = np.concatenate(
        [conf_permeability_xy_cells, permeability_xy_cells, conf_permeability_xy_cells], axis = 2)
    tot_permeability_z = np.concatenate(
        [conf_permeability_z_cells, permeability_z_cells, conf_permeability_z_cells], axis=2)
    tot_porosity = np.concatenate(
        [conf_porosity_cells, porosity_cells, conf_porosity_cells], axis = 2)
    tot_heat_capacity = np.concatenate(
        [conf_heat_capacity_cells, heat_capacity_cells, conf_heat_capacity_cells], axis = 2)
    tot_t_cond = np.concatenate(
        [conf_thermal_conductivity_cells, thermal_conductivity_cells, conf_thermal_conductivity_cells], axis = 2)

    tot_actnum = np.concatenate(
        [act_num0, act_num1, act_num0], axis=2)

    # Check shape
    print(nx_full, ny_full)
    print(np.shape(tot_permeability_xy))

    geomodel = xr.Dataset(
        data_vars={
            "permeability_xy": (("x", "y", "z"), tot_permeability_xy),
            "permeability_z":  (("x", "y", "z"), tot_permeability_z),
            "porosity":        (("x", "y", "z"), tot_porosity),
            "heat_capacity":   (("x", "y", "z"), tot_heat_capacity),
            "thermal_conductivity": (("x", "y", "z"), tot_t_cond),
            "actnum":          (("x", "y", "z"), tot_actnum)},

        coords={
            "x": x_cell_centers,
            "y": y_cell_centers,
            "z": z_center},

        attrs={
            "hwx" : hwx,
            "hwy" : hwy,
            "dx": dx_full,
            "dy": dy_full,
            "dz": dz_full,
            "nly_top" : nz_conf,
            "nly_res" : nz_aq,
            "nly_bot" : nz_conf,
            "grid_type": "Varying"})

    print("Hot well x- and y-coordinate:", x_cell_centers[hwx], y_cell_centers[hwy])
    return geomodel

def run_DARTS (simulation_name,
               geomodel,
               nyears,
               Tin, TCutOff,
               volumetric_rate, operational_profile, storage_periods,
               set_transition_runtime = 1e-3,
               well_diameter = 1,
               n_points = 256):

    ### =========== Simulation settings ==============
    # ------------- Output directory ----------------
    output_directory = f"3D_{simulation_name}"  # Spatial output
    output_h5 = f"h5_{simulation_name}"  # DARTS output folder

    os.makedirs(output_h5, exist_ok=True)
    output_folder = output_h5

    output_well_data_excel = f"WD_{simulation_name}.xlsx"

    os.makedirs(output_directory, exist_ok=True)

    # ------------- Input geomodel -------------------
    perm_h = geomodel['permeability_xy'].values
    perm_v = geomodel['permeability_z'].values
    poro = geomodel['porosity'].values
    hcap = geomodel['heat_capacity'].values
    tcond = geomodel['thermal_conductivity'].values
    actnum =geomodel["actnum"].values

    # Load XY-plane well indices
    hwx = geomodel.attrs['hwx']
    hwy = geomodel.attrs['hwy']

    dX_array = geomodel.attrs['dx']
    dY_array = geomodel.attrs['dy']
    dZ_array = geomodel.attrs['dz']

    nly_top = geomodel.attrs["nly_top"]
    nly_res = geomodel.attrs["nly_res"]
    nly_bot = geomodel.attrs["nly_bot"]

    # --------------- Initial conditions reservoir -------
    geothermal_grad = 0 #(K / km), the geothermal gradient for the initial condition of the reservoir

    # -------------- Operational parameters -----------------------
    # Well temperatures to Kelvin
    InjT = 273.15 + Tin # (K)
    #TCutOff = 273.15 + tcuo # (K), Serves as the injection temperature of the warm well

    # --------------- Simulation time and space settings  --------
    set_run_years = nyears #Total simulation length (years)
    max_ts = 1 #Maximum time step size (days)
    dt_mult = 8 #Time step upscaling (-)
    #set_transition_runtime = 1 #dt after an operational period change [days] #1e-3

    depth_to_top = 0

    ### ============== Run Simulation =====================
    #Input model params here
    m = Model(dX_array, dY_array, dZ_array,
              nly_top, nly_res, nly_bot,
              perm_h, perm_v, poro, hcap, tcond,
              actnum,
              hwx, hwy, well_diameter,
              depth_to_top, geothermal_grad,
              dt_mult, max_ts,
              n_points = n_points)
    m.init()
    m.set_output(output_folder = output_folder)

    iterr = 1
    for k in range(set_run_years):
        for i, runtime in enumerate(operational_profile):
            if storage_periods[i] == 'Charge':

                m.set_rate_hot(volumetric_rate, temp=InjT, func='inj')
                m.set_rate_cold(-1 * volumetric_rate, func='prod')
                print('Operation: Charge')

            elif storage_periods[i] == 'Discharge':

                m.set_rate_hot(-1 * volumetric_rate, func='prod')
                m.set_rate_cold(volumetric_rate, temp=TCutOff, func='inj')
                print('Operation: Discharge')

            elif storage_periods[i] == 'Rest':

                m.set_rate_hot(0, func='prod')
                m.set_rate_cold(0, func='prod')
                print('Operation: Rest')

            m.run(runtime, restart_dt=set_transition_runtime)
            print("\nIterr :", iterr, "\tYear :", k, "\tRun Time :", runtime)
            print("\n")
            iterr += 1
    m.print_timers()
    m.print_stat()

    # Comment the following 2 lines to ensure no 3D output data is generated
    output_props = ['temperature', 'pressure']
    m.output.output_to_vtk(output_properties=output_props, output_directory = output_directory)


    # %%-----------------Write Results to Excel-----------------
    # output well information to Excel file
    td = pd.DataFrame.from_dict(m.physics.engine.time_data)
    #writer = pd.ExcelWriter(output_well_data_excel)
    #td.to_excel(writer, 'Sheet1')
    #writer.close()
    with pd.ExcelWriter(output_well_data_excel) as writer:
        td.to_excel(writer, sheet_name='Sheet1')

    # %%-----------------read H5-----------------
    #well_id, well_depth = write_well_perforation_id(m)
    #r = read_well_h5(well_block_id=well_id, well_block_depth=well_depth)
    #r.draw_combined_well_data()  # All wells on same subplots
