import pathlib
import time
from vkoga.kernels import Gaussian

from ml_control.completely_reduced_model import CompletelyReducedModel
from ml_control.logger import getLogger
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.problem_definitions.heat_equation import create_heat_equation_problem_complex
from ml_control.reduced_model import ReducedModel
from ml_control.reduction_procedures import setup_initial_system_bases

from adaptive_ml_control.extended_adaptive_model_hierarchy import ExtendedAdaptiveModelHierarchy


full_model = create_heat_equation_problem_complex()
parameter_space = full_model.parameter_space

num_parameters = 100
training_parameters = parameter_space.sample_uniformly(num_parameters)

rb_rom = ReducedModel([], full_model)
reduction_strategy = "inc_hapod"
reduction_strategy_parameters = {"primal": {"steps": 50, "eps": 1e-8, "omega": 0.9},
                                 "adjoint": {"steps": 50, "eps": 1e-8, "omega": 0.9}}

initial_parameter = [1.5, 1.]
Vpr, Wpr, Vad, Wad = setup_initial_system_bases(full_model, initial_parameter, reduction_strategy,
                                                reduction_strategy_parameters)
c_rom = CompletelyReducedModel(Vpr, Wpr, Vad, Wad, [], full_model, reduction_strategy=reduction_strategy,
                               reduction_strategy_parameters=reduction_strategy_parameters)
zero_padding = False
ml_c_rom = KernelReducedModel(c_rom, [], zero_padding=zero_padding)

tol = 1e-4
ml_training_parameters = {"kernel": Gaussian(0.5), "tol_p": 1e-10}
adaptive_model = ExtendedAdaptiveModelHierarchy(full_model, rb_rom, c_rom, ml_c_rom, tol=tol,
                                                ml_training_parameters=ml_training_parameters)

logger = getLogger("HeatEquation", level="INFO")

logger.info(f"Solving for {len(training_parameters)} parameters ...")
for i, mu in enumerate(training_parameters):
    with logger.block(f"Parameter number {i} ..."):
        adaptive_model.solve(mu)

adaptive_model.print_summary()
timestr = time.strftime("%Y%m%d-%H%M%S")
filepath = f"results_heat_equation_{timestr}/"
pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
adaptive_model.write_results_to_file(filepath)
plt = adaptive_model.plot_detailed_timings()
plt.show()
adaptive_model.write_detailed_timings_as_tex(filepath + "detailed_timings.tex")
plt.close()
plt = adaptive_model.plot_detailed_error_estimation()
plt.show()
adaptive_model.write_detailed_error_estimation_as_tex(filepath + "detailed_error_estimations.tex")
plt.close()
