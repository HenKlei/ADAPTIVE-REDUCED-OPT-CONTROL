import pathlib
import time
from vkoga.kernels import Wendland

from ml_control.logger import getLogger
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.machine_learning_models.neural_network_reduced_model import NeuralNetworkReducedModel
from ml_control.problem_definitions.cookies import create_cookies_problem
from ml_control.reduced_model import ReducedModel

from adaptive_ml_control.adaptive_model_hierarchy import AdaptiveModelHierarchy


nt = 50
full_model = create_cookies_problem(nt=nt, problem_size="medium")
parameter_space = full_model.parameter_space

num_parameters = 10000
training_parameters = parameter_space.sample_randomly(num_parameters)

rb_rom = ReducedModel([], full_model)

ml_rom = "kernel"

if ml_rom == "kernel":
    zero_padding = False
    ml_rom = KernelReducedModel(rb_rom, [], zero_padding=zero_padding, scale_inputs=True)
    ml_training_parameters = {"kernel": Wendland(ep=0.1, k=1, d=2), "greedy_type": "f_greedy", "tol_f": 1e-10}
elif ml_rom == "neural_network":
    zero_padding = True
    ml_rom = NeuralNetworkReducedModel(rb_rom, [], zero_padding=zero_padding, scale_inputs=True)
    ml_training_parameters = {}
    # TODO: Implement log scaling of inputs?!
else:
    raise NotImplementedError

tol = 1e-4
adaptive_model = AdaptiveModelHierarchy(full_model, rb_rom, ml_rom, tol=tol,
                                        ml_training_parameters=ml_training_parameters, ml_training_frequency=20)

logger = getLogger("CookiesProblem", level="INFO")

logger.info(f"Solving for {len(training_parameters)} parameters ...")
for i, mu in enumerate(training_parameters):
    with logger.block(f"Parameter number {i} ..."):
        adaptive_model.solve(mu)

adaptive_model.print_summary()
timestr = time.strftime("%Y%m%d-%H%M%S")
filepath = f"results_cookies_{timestr}/"
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
