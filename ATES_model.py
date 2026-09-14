from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from dartsflash.mixtures import DARTSFlash, CompData, EoS, IAPWS
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
# Needed for the original workings of DARTS with better viscosity evaluation
#from darts.physics.properties.viscosity import MaoDuan2009

from darts.models.darts_model import DartsModel
from darts.engines import redirect_darts_output, well_control_iface
from darts.physics.properties.eos_properties import EoSEnthalpy
from darts.physics.properties.flash import SinglePhase
from darts.reservoirs.struct_reservoir import StructReservoir
from dartsflash.components import CompData
from darts.nonlinear_solvers import NewtonSolver, ChopSpec

redirect_darts_output('LogFile_Run_HT_ATES_DELFT.log')

from darts.engines import set_num_threads

set_num_threads(4)

import numpy as np

class Sharqawy2012:
    """
    Water density correlation.
    """

    def evaluate(self, pressure, temperature, x):
        T = temperature - 273.15

        rho = 1000.0 - ((T - 4.0) ** 2) / 207.0

        return rho


class Voss1984:
    """
    Water viscosity correlation.
    """

    def evaluate(self, pressure, temperature, x, rho):
        T = temperature - 273.15

        mu = 2.394e-5 * (10.0 ** (248.37 / (T + 133.15)))

        return mu

# %%
class Model(DartsModel):

    def __init__(self,
                 dx_array, dy_array, dz_array,
                 nly_top, nly_res, nly_bot,
                 permXYZ_h, permXYZ_v, poroXYZ, hcapXYZ, tcondXYZ,
                 actnum,
                 hwx, hwy, well_diameter,
                 depth_to_top, geothermal_grad,
                 ts_mult, ts_max,
                 n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()
        self.platform = 'cpu'
        # #------------Set Node Number------------
        self.nly_top = int(nly_top)  # layer number in cap rock
        self.nly_res = int(nly_res)  # int(n_ly[1]) # layer number in reservoir rock
        self.nly_bot = int(nly_bot)  # layer number in bottom formation

        nx = len(dx_array)
        print("nx basic ates", nx)
        ny = len(dy_array)
        print("ny basic ates", ny)
        nz = len(dz_array)
        print("nz basic ates", nz)

        self.hwx = hwx
        self.hwy = hwy
        self.well_diameter = well_diameter

        self.ohwx = hwx + 10 #observation well at 10 meters apart
        self.ohwy = hwy

        perm_h = permXYZ_h.ravel(order = 'F')
        perm_v = permXYZ_v.ravel(order= 'F')
        poro = poroXYZ.ravel(order= 'F')
        hcap = hcapXYZ.ravel(order= 'F')
        tcond = tcondXYZ.ravel(order = 'F')
        actnum_F = actnum.ravel(order = 'F')

        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz,
                                         dx=dx_array, dy=dy_array, dz=dz_array,
                                         permx=perm_h, permy=perm_h, permz=perm_v,
                                         poro=poro, start_z=depth_to_top, hcap=hcap, rcond=tcond,
                                         actnum = actnum_F)

        self.reservoir.boundary_volumes['yz_minus'] = 1e20
        self.reservoir.boundary_volumes['yz_plus'] = 1e20
        self.reservoir.boundary_volumes['xz_minus'] = 1e20
        self.reservoir.boundary_volumes['xz_plus'] = 1e20
        self.reservoir.boundary_volumes['xy_minus'] = 1e20
        self.reservoir.boundary_volumes['xy_plus'] = 1e20

        self.geothermal_grad = geothermal_grad

        # Pre-defined physics single phase, single component
        self.set_physics_super(zero=1e-12, n_points=n_points, components=["H2O"])

        self.ts_control.dt_first = 1e-12 #days
        self.ts_control.dt_mult = 4 #days
        self.ts_control.dt_max = 1 #days
        self.ts_control.runtime = 5 #days

        # Solver settings
        super().set_solver() # Check if I need this
        self.nonlinear_solver.spec.tolerance = 1e-3
        self.nonlinear_solver.spec.max_iterations = 20
        self.linear_solver.spec.tolerance = 1e-6
        self.linear_solver.spec.max_iterations = 50

        # Try both below to see which one works
        self.nonlinear_solver.spec.chop = ChopSpec(mode='global', factor=1.0)
        #self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=20, chop=ChopSpec(mode='global', factor=1.0))

        self.timer.node["initialization"].stop()
        # Check if I need it, maybe useful
        self.print_config()

    def set_physics_super(self, zero, n_points, components):
        """Physical properties"""
        # Fluid components, ions and solid

        phases = ["L"] # used to be ["Aq"]
        comp_data = CompData(components, setprops=True)
        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               rock_comp=1e-5, eps_z=zero / 10)

        # Single phase, single component bookkeeping Flash
        flash_ev = SinglePhase(nc=len(components))
        property_container.flash_ev = flash_ev

        # Enthalpy evaluation using IAPWS
        iapws = IAPWS(iapws_ideal=True, ice_phase=False)
        iapws.init_flash(flash_type=DARTSFlash.FlashType.PTFlash)

        property_container.enthalpy_ev = {"L": EoSEnthalpy(eos=iapws.eos["IAPWS"], root_flag=EoS.RootFlag.MIN)}

        ### Before modification
        # property_container.flash_ev = NegativeFlash2(flash_params)
        # property_container.flash_ev = SinglePhase(nc=2)

        ### Locally defined density & viscosity relations
        property_container.density_ev = {'L': Sharqawy2012()}
        property_container.viscosity_ev = {'L': Voss1984()}

        ### DARTS OG density & viscosity relations
        #property_container.density_ev = {'L', Garcia2001(components)}
        ### Update to flash-IAPWS formulation
        #property_container.density_ev = {'L': EoSDensity(flash.eos["IAPWS"], comp_data.Mw, EoS.RootFlag.MIN)}
        #property_container.viscosity_ev = {'L', MaoDuan2009(components)}

        property_container.rel_perm_ev = {'L': PhaseRelPerm("wat", swc=0.0)}

        property_container.conductivity_ev = {'L': ConstFunc(172.8)} # kJ/m/day/K
        property_container.capillary_pressure_ev = {'L': ConstFunc(0.)}


        p_min = 0.01
        p_max = 50.0
        T_min = 273.15
        T_max = 400.0

        p_step = (p_max - p_min) / (n_points - 1)
        T_step = (T_max - T_min) / (n_points - 1)

        self.physics = PhysicsBase(
            components,
            phases,
            self.timer,
            axes_step=[p_step, T_step],
            axes_origin=[p_min, T_min],
            epsilon_z=zero / 10,
            state_spec=PhysicsBase.StateSpecification.PT,
            cache=False
        )

        #self.physics = Compositional(components, phases, self.timer, n_points, min_p=0.01, max_p=50, min_z=zero / 10,
                                     #max_z=1 - zero / 10, min_t=273.15 + 5, max_t=400, epsilon_z= zero/10,
                                     #state_spec=Compositional.StateSpecification.PT, cache=False)
        #self.physics.thermal = thermal
        self.physics.add_property_region(property_container)
        self.physics.init_physics()

        return

    def set_wells(self):
        top_ind = self.nly_top + 1
        btm_ind = self.nly_top + self.nly_res #self.n_ly_cap + self.n_ly_res #If I add 2 confining layers

        well_name = "H_1"

        self.reservoir.add_well(well_name)

        for k in range(top_ind, btm_ind + 1):
            self.reservoir.add_perforation(
                well_name,
                res_cell_idx=(self.hwx, self.hwy, k),
                well_diameter = self.well_diameter,
                verbose=True,
                well_indexD=0,
                skin = 10, #Important should I do anything with the skinnfactor? I am not really looking at production data anyway
                ms_epm = False)

    def set_initial_conditions(self):
        input_depth = [0., np.amax(self.reservoir.mesh.depth)]
        input_distribution = {'pressure': [1., 1. + input_depth[1] * 100. / 1000],
                              'temperature': [285.15, 285.15 + input_depth[1] * self.geothermal_grad / 1000]}
        return self.physics.set_initial_conditions_from_depth_table(self.reservoir.mesh,
                                                                    input_distribution=input_distribution,
                                                                    input_depth=input_depth)
    def set_well_controls(self, h_func=None, l_func=None):
        for i, w in enumerate(self.reservoir.wells):
            if 'H' in w.name:
                self.physics.set_well_controls(wctrl=w.control,control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=0., phase_name='L', inj_composition=[],
                                               inj_temp=273.15 + 90)
            elif 'C' in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=False, target=0., phase_name='L')

            elif 'O' in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=False, target=0., phase_name='L')

    def set_rate_hot(self, rate, temp=300, func='inj'):
        for w in self.reservoir.wells:
            if 'H' in w.name:
                if func == 'inj':
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=True, target=rate, phase_name='L', inj_composition=[],
                                                   inj_temp=temp)
                    # w.constraint = self.physics.new_bhp_water_inj(self.midrespress + self.bhp_limit, temp)
                if func == 'prod':
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=False, target=rate, phase_name='L')
                    # w.constraint = self.physics.new_bhp_prod(self.midrespress - self.bhp_limit)

    def set_rate_cold(self, rate, temp=300, func='inj'):
        # w = self.reservoir.wells[welln]
        for w in self.reservoir.wells:
            if 'C' in w.name:
                if func == 'inj':
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=True, target=rate, phase_name='L', inj_composition=[],
                                                   inj_temp=temp)
                    # w.constraint = self.physics.new_bhp_water_inj(self.midrespress + self.bhp_limit, temp)
                if func == 'prod':
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=False, target=rate, phase_name='L')
                    # w.constraint = self.physics.new_bhp_prod(self.midrespress - self.bhp_limit)

