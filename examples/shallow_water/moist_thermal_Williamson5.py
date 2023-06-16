"""
Moist flow over a mountain test case from Zerroukat and Allen, using their moist
thermal shallow water model and moist physics scheme.
"""
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, exp, conditional, cos,
                       acos, Constant, Function, File)

# ----------------------------------------------------------------- #
# Test case parameters
# ----------------------------------------------------------------- #

day = 24*60*60
dt = 300
tmax = 50*day
R = 6371220.
H = 5960.
u_max = 20.
ndumps = 50
dumpfreq = int(tmax / (ndumps*dt))
# moist shallow water parameters
epsilon = 1/300
SP = -40*epsilon
EQ = 30*epsilon
NP =-20*epsilon
mu1 = 0.05
mu2 = 0.98
L = 10
q0 = 0.0492238 # (from Ferguson and Jablonowski AMR paper)
# q0 = 135
beta2 = 1
qprecip = 10e-4
gamma_r = 10e-3
# topography parameters
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
phi_c = pi/6.

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=4, degree=1)
degree = 1
domain = Domain(mesh, dt, "BDM", degree)
x = SpatialCoordinate(mesh)

# Equation
parameters = ShallowWaterParameters(H=H)
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R

# topography
phi, lamda = latlon_coords(mesh)
lsq = (lamda - lamda_c)**2
thsq = (phi - phi_c)**2
rsq = min_value(R0sq, lsq+thsq)
r = sqrt(rsq)
tpexpr = 2000 * (1 - r/R0)

tracers = [WaterVapour(space='DG'), CloudWater(space='DG'), Rain(space='DG')]
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=tpexpr,
                             thermal=True,
                             active_tracers=tracers)

# I/O
dirname = "ZA_moist_thermal_williamson5_temp"
output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D'],
                          dumpfreq=dumpfreq,
                          log_level='INFO')
diagnostic_fields = [Sum('D', 'topography'), CourantNumber()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Saturation function
def sat_func(x_in):
    h = x_in.split()[1]
    b = x_in.split()[2]
    return (q0/(g*h + g*tpexpr)) * exp(20*(1 - b/g))

# Feedback proportionality is dependent on h and b
def gamma_v(x_in):
    h = x_in.split()[1]
    b = x_in.split()[2]
    return (1 + L*(20*q0/(g*h + g*tpexpr) * exp(20*(1 - b/g))))**(-1)


ReversibleAdjustment(eqns, sat_func, L, time_varying_saturation=True,
                     parameters=parameters, thermal_feedback=True,
                     beta2=beta2, gamma_v=gamma_v,
                     time_varying_gamma_v=True)

InstantRain(eqns, qprecip, vapour_name="cloud_water", rain_name="rain",
            gamma_r=gamma_r)

# Time stepper
stepper = Timestepper(eqns, RK4(domain), io)

# initial conditions
u0 = stepper.fields("u")
D0 = stepper.fields("D")
b0 = stepper.fields("b")
v0 = stepper.fields("water_vapour")
c0 = stepper.fields("cloud_water")
r0 = stepper.fields("rain")

uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])

g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - tpexpr

# expression for initial buoyancy
F = (2/(pi**2))*(phi*(phi-pi/2)*SP - 2*(phi+pi/2)*(phi-pi/2)*(1-mu1)*EQ + phi*(phi+pi/2)*NP)
theta_expr = F + mu1*EQ*cos(phi)*sin(lamda)
bexpr = g * (1 - theta_expr)

# write out files to look at initial conditions
ICs_file = File("outfile.pvd")
theta_func = Function(b0.function_space()).interpolate(theta_expr)
F_func = Function(b0.function_space()).interpolate(F)
diurnal_func = Function(b0.function_space()).interpolate(mu1*EQ*cos(phi)*sin(lamda))
ICs_file.write(theta_func, F_func, diurnal_func)
theta_field = stepper.fields( "theta", space=b0.function_space(), dump=True)
theta_field.interpolate(theta_expr)

# expression for initial vapour depends on initial saturation
initial_msat = q0/(g*D0 + g*tpexpr) * exp(20*theta_expr)
vexpr = mu2 * initial_msat

# initialise (cloud and rain initially zero)
u0.project(uexpr)
D0.interpolate(Dexpr)
b0.interpolate(bexpr)
v0.interpolate(vexpr)

# ----------------------------------------------------------------- #
# Run
# ----------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
