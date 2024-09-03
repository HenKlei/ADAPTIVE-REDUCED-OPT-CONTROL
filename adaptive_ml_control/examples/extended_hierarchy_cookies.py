import pathlib
import time
from vkoga.kernels import Wendland

from ml_control.completely_reduced_model import CompletelyReducedModel
from ml_control.logger import getLogger
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.problem_definitions.cookies import create_cookies_problem
from ml_control.reduced_model import ReducedModel
from ml_control.reduction_procedures import setup_initial_system_bases

from adaptive_ml_control.extended_adaptive_model_hierarchy import ExtendedAdaptiveModelHierarchy


nt = 50
full_model = create_cookies_problem(nt=nt, problem_size="medium")
parameter_space = full_model.parameter_space

num_parameters = 10000
training_parameters = parameter_space.sample_randomly(num_parameters)

rb_rom = ReducedModel([], full_model)
reduction_strategy = "dist_hapod"
reduction_strategy_parameters = {"primal": {"num_slices": 50, "eps": 1e-9, "omega": 0.9},
                                 "adjoint": {"num_slices": 50, "eps": 1e-9, "omega": 0.9}}

initial_parameter = [1, 1]
Vpr, Wpr, Vad, Wad, svals_pr, svals_ad = setup_initial_system_bases(full_model, initial_parameter, reduction_strategy,
                                                                    reduction_strategy_parameters)
c_rom = CompletelyReducedModel(Vpr, Wpr, Vad, Wad, [], full_model, reduction_strategy=reduction_strategy,
                               reduction_strategy_parameters=reduction_strategy_parameters,
                               svals_pr=svals_pr, svals_ad=svals_ad)
zero_padding = False
ml_c_rom = KernelReducedModel(c_rom, [], zero_padding=zero_padding, scale_inputs=True)

tol = 1e-4
zero_padding = False
ml_training_parameters = {"kernel": Wendland(ep=0.1, k=1, d=2), "greedy_type": "f_greedy", "tol_f": 1e-10}
adaptive_model = ExtendedAdaptiveModelHierarchy(full_model, rb_rom, c_rom, ml_c_rom, tol=tol,
                                                ml_training_parameters=ml_training_parameters,
                                                ml_training_frequency=20)

logger = getLogger("CookiesProblemExt", level="INFO")

logger.info(f"Solving for {len(training_parameters)} parameters ...")
for i, mu in enumerate(training_parameters):
    with logger.block(f"Parameter number {i} ..."):
        adaptive_model.solve(mu)

adaptive_model.print_summary()
timestr = time.strftime("%Y%m%d-%H%M%S")
if ml_c_rom is None:
    filepath = f"results_cookies_extended_without_ml_{timestr}/"
else:
    filepath = f"results_cookies_extended_with_ml_{timestr}/"
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
