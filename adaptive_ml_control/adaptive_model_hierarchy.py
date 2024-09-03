import numpy as np
import time

from ml_control.logger import getLogger


class AdaptiveModelHierarchy:
    def __init__(self, fom, rb_rom, ml_rom, tol=1e-4, ml_training_frequency=10, ml_training_parameters={}):
        self.fom = fom
        self.rb_rom = rb_rom
        self.ml_rom = ml_rom

        self.tol = tol

        self.ml_training_frequency = ml_training_frequency
        self.ml_training_parameters = ml_training_parameters

        self.logger = getLogger("AdaptiveModelHierarchy", level='INFO')

        self.counter = {"fom": 0, "rb_rom": 0, "ml_rom": 0, "rb_rom_est": 0, "ml_rom_est": 0,
                        "rb_rom_extension": 0, "ml_rom_training": 0}
        self.timings = {"fom": 0., "rb_rom": 0., "ml_rom": 0., "rb_rom_est": 0., "ml_rom_est": 0.,
                        "rb_rom_extension": 0., "ml_rom_training": 0.}
        self.detailed_results = []

    def _summary(self, func, postfix=""):
        func("Full order model:" + postfix)
        func("=================" + postfix)
        self.fom._summary(func, postfix=postfix)

        total_num_parameters = self.counter["fom"] + self.counter["rb_rom"] + self.counter["ml_rom"]
        func(f"Total number of solves: {total_num_parameters}" + postfix)
        func("" + postfix)
        func("Number of solves per model:" + postfix)
        func(f"FOM: {self.counter['fom']}" + postfix)
        func(f"RB-ROM: {self.counter['rb_rom']}" + postfix)
        func(f"ML-ROM: {self.counter['ml_rom']}" + postfix)
        func("" + postfix)
        func("Number of error estimations per model:" + postfix)
        func(f"RB-ROM: {self.counter['rb_rom_est']}" + postfix)
        func(f"ML-ROM: {self.counter['ml_rom_est']}" + postfix)
        func("" + postfix)
        func("Total time per model (error estimation and solving):" + postfix)
        func(f"FOM: {self.timings['fom']}" + postfix)
        func(f"RB-ROM: {self.timings['rb_rom'] + self.timings['rb_rom_est']}" + postfix)
        func(f"ML-ROM: {self.timings['ml_rom'] + self.timings['ml_rom_est']}" + postfix)
        func("" + postfix)
        func("Average time per model (solve and error estimation) per solve:" + postfix)
        if self.counter["fom"] > 0:
            func(f"FOM: {self.timings['fom'] / self.counter['fom']}" + postfix)
        if self.counter["rb_rom"] > 0:
            func(f"RB-ROM: {(self.timings['rb_rom'] + self.timings['rb_rom_est']) / self.counter['rb_rom']}" + postfix)
        if self.counter["ml_rom"] > 0:
            func(f"ML-ROM: {(self.timings['ml_rom'] +self.timings['ml_rom_est']) / self.counter['ml_rom']}" + postfix)
        func("" + postfix)
        func("Average time per error estimation per model:" + postfix)
        if self.counter["rb_rom_est"] > 0:
            func(f"RB-ROM: {self.timings['rb_rom_est'] / self.counter['rb_rom_est']}" + postfix)
        if self.counter["ml_rom_est"] > 0:
            func(f"ML-ROM: {self.timings['ml_rom_est'] / self.counter['ml_rom_est']}" + postfix)
        func("" + postfix)
        func("Average time per solve per model (reuses information from error estimation):" + postfix)
        if self.counter["fom"] > 0:
            func(f"FOM: {self.timings['fom'] / self.counter['fom']}" + postfix)
        if self.counter["rb_rom"] > 0:
            func(f"RB-ROM: {self.timings['rb_rom'] / self.counter['rb_rom']}" + postfix)
        if self.counter["ml_rom"] > 0:
            func(f"ML-ROM: {self.timings['ml_rom'] / self.counter['ml_rom']}" + postfix)
        func("" + postfix)
        func("Total time of reduced model training steps:" + postfix)
        func(f"RB-ROM: {self.timings['rb_rom_extension']}" + postfix)
        func(f"ML-ROM: {self.timings['ml_rom_training']}" + postfix)
        func("" + postfix)
        func("Total number of reduced model updates:" + postfix)
        func(f"RB-ROM: {self.counter['rb_rom_extension']}" + postfix)
        func(f"ML-ROM: {self.counter['ml_rom_training']}" + postfix)

    def print_summary(self):
        with self.logger.block("Summary of the results of the adaptive model hierachy:"):
            self._summary(self.logger.info)

    def write_results_to_file(self, filepath):
        with open(filepath + "summary.txt", "a") as f:
            self._summary(f.write, postfix="\n")
        with open(filepath + "detailed_results.txt", "a") as f:
            f.write("Operation\tRequired time\tEstimated error")
            for operation, res in self.detailed_results:
                if operation.endswith("_est"):
                    f.write(f"{operation}\t{res[0]}\t{res[1]}\n")
                else:
                    f.write(f"{operation}\t{res}\n")

    def plot_detailed_timings(self):
        total_num_parameters = self.counter["fom"] + self.counter["rb_rom"] + self.counter["ml_rom"]
        operations_per_evaluation = {"ml_rom": np.zeros(total_num_parameters),
                                     "rb_rom": np.zeros(total_num_parameters),
                                     "fom": np.zeros(total_num_parameters),
                                     "rb_rom_extension": np.zeros(total_num_parameters),
                                     "ml_rom_training": np.zeros(total_num_parameters)}
        labels = {"ml_rom": "ML-ROM",
                  "rb_rom": "RB-ROM",
                  "fom": "FOM",
                  "rb_rom_extension": "Extend RB-ROM",
                  "ml_rom_training": "Train ML-ROM"}
        from matplotlib.colors import to_rgba
        timings_colors = {"fom": to_rgba('#74a9cf', 1.),
                          "rb_rom": to_rgba('#fd8d3c', 1.),
                          "rb_rom_extension": to_rgba('#fecc5c', 1.),
                          "ml_rom": to_rgba('#006837', 1.),
                          "ml_rom_training": to_rgba('#78c679', 1.)}

        i = -1
        for operation, res in self.detailed_results:
            if operation == "ml_rom_est":
                i += 1
            if operation.endswith("_est"):
                operation = operation[:-4]
                required_time = res[0]
            else:
                required_time = res
            operations_per_evaluation[operation][i] += required_time

        import matplotlib.pyplot as plt
        plt.figure()
        bottom = np.zeros(total_num_parameters)

        for operation, required_time in operations_per_evaluation.items():
            plt.bar(np.arange(total_num_parameters)[required_time > 0.], required_time[required_time > 0.], 1.,
                    label=labels[operation], bottom=bottom[required_time > 0.], color=timings_colors[operation])
            bottom += required_time

        plt.xticks(np.arange(total_num_parameters), np.arange(1, total_num_parameters + 1, 1, dtype=int))
        plt.xlabel("Number of evaluation")
        plt.ylabel("Required time in seconds")
        plt.title("Evaluations of the different models with timings")

        return plt

    def write_detailed_timings_as_tex(self, filepath):
        _ = self.plot_detailed_timings()
        import tikzplotlib
        tikzplotlib.save(filepath)

    def plot_detailed_error_estimation(self):
        total_num_parameters = self.counter["fom"] + self.counter["rb_rom"] + self.counter["ml_rom"]
        estimated_error_per_evaluation = {"ml_rom_est": ([], []),
                                          "rb_rom_est": ([], [])}
        labels = {"ml_rom_est": "ML-ROM",
                  "rb_rom_est": "RB-ROM"}
        from matplotlib.colors import to_rgba
        estimations_colors = {"rb_rom_est": to_rgba('#fd8d3c', 1.),
                              "ml_rom_est": to_rgba('#006837', 1.)}

        i = -1
        for operation, res in self.detailed_results:
            if operation == "ml_rom_est":
                i += 1
            if operation.endswith("_est"):
                estimated_error = res[1]
                estimated_error_per_evaluation[operation][0].append(i)
                estimated_error_per_evaluation[operation][1].append(estimated_error)

        import matplotlib.pyplot as plt
        plt.figure()

        for operation, res in estimated_error_per_evaluation.items():
            plt.plot(res[0], res[1], label=labels[operation], color=estimations_colors[operation])

        plt.xticks(np.arange(total_num_parameters), np.arange(1, total_num_parameters + 1, 1, dtype=int))
        plt.xlabel("Number of evaluation")
        plt.ylabel("Estimated error of the respective model")
        plt.title("Evaluations of the different models with error estimates")

        return plt

    def write_detailed_error_estimation_as_tex(self, filepath):
        _ = self.plot_detailed_error_estimation()
        import tikzplotlib
        tikzplotlib.save(filepath)

    def solve(self, mu):
        with self.logger.block(f"Solving for mu={mu} ..."):
            self.logger.info("Evaluating error of the machine learning ...")
            # Estimate error of machine learning
            tic = time.perf_counter()
            coeffs_ml = self.ml_rom.solve(mu)
            ml_rom_est_error = self.ml_rom.estimate_error(mu, coeffs_ml)
            ml_rom_estimation_time = time.perf_counter() - tic
            self.logger.info(f"Estimated machine learning error is {ml_rom_est_error:.3e} ...")
            self.counter["ml_rom_est"] += 1
            self.timings["ml_rom_est"] += ml_rom_estimation_time
            self.detailed_results.append(("ml_rom_est", (ml_rom_estimation_time, ml_rom_est_error)))
            # Try machine learning surrogate first
            if ml_rom_est_error <= self.tol:
                self.logger.info("Machine learning sufficiently accurate, returning solution ...")
                tic = time.perf_counter()
                u_ml = self.ml_rom.compute_control(mu, coeffs_ml)
                ml_rom_time = time.perf_counter() - tic
                self.counter["ml_rom"] += 1
                self.timings["ml_rom"] += ml_rom_time
                self.detailed_results.append(("ml_rom", ml_rom_time))
                # Return solution
                return u_ml

            self.logger.info("Evaluating error of the reduced basis reduced model ...")
            # Estimate error of reduced basis model
            tic = time.perf_counter()
            coeffs_rb = self.rb_rom.solve(mu)
            rb_rom_est_error = self.rb_rom.estimate_error(mu, coeffs_rb)
            rb_rom_estimation_time = time.perf_counter() - tic
            self.logger.info(f"Estimated reduced basis reduced model error is {rb_rom_est_error:.3e} ...")
            self.counter["rb_rom_est"] += 1
            self.timings["rb_rom_est"] += rb_rom_estimation_time
            self.detailed_results.append(("rb_rom_est", (rb_rom_estimation_time, rb_rom_est_error)))
            # Fall back to the reduced basis model if possible
            if rb_rom_est_error <= self.tol:
                self.logger.info("Reduced basis reduced model sufficiently accurate, extending training data and "
                                 "returning solution ...")
                tic = time.perf_counter()
                u_rb = self.rb_rom.compute_control(mu, coeffs_rb)
                rb_rom_time = time.perf_counter() - tic
                self.counter["rb_rom"] += 1
                self.timings["rb_rom"] += rb_rom_time
                self.detailed_results.append(("rb_rom", rb_rom_time))
                self.ml_rom.training_data.append((mu, coeffs_rb))
                # Train machine learning if desired
                if len(self.ml_rom.training_data) % self.ml_training_frequency == 0:
                    tic = time.perf_counter()
                    self.ml_rom.train(**self.ml_training_parameters)
                    ml_rom_training_time = time.perf_counter() - tic
                    self.counter["ml_rom_training"] += 1
                    self.timings["ml_rom_training"] += ml_rom_training_time
                    self.detailed_results.append(("ml_rom_training", ml_rom_training_time))
                # Return solution
                return u_rb

            self.logger.info("Falling back to the full model ...")
            # Fall back to the full order model
            tic = time.perf_counter()
            phi_fom = self.fom.solve(mu)
            u_fom = self.fom.compute_control(mu, phi_fom)
            fom_time = time.perf_counter() - tic
            self.counter["fom"] += 1
            self.timings["fom"] += fom_time
            self.detailed_results.append(("fom", fom_time))

            self.logger.info("Extending the reduced basis and adding zero padding to the training data ...")
            # Extend reduced basis
            tic = time.perf_counter()
            self.rb_rom.extend(mu, phi_fom)
            rb_rom_extension_time = time.perf_counter() - tic
            self.counter["rb_rom_extension"] += 1
            self.timings["rb_rom_extension"] += rb_rom_extension_time
            self.detailed_results.append(("rb_rom_extension", rb_rom_extension_time))

            self.ml_rom.extend_model()

            # Training machine learning model again after extending the reduced basis
            if (len(self.ml_rom.training_data) > 0
                    and len(self.ml_rom.training_data) % self.ml_training_frequency == 0):
                tic = time.perf_counter()
                self.ml_rom.train(**self.ml_training_parameters)
                ml_rom_training_time = time.perf_counter() - tic
                self.counter["ml_rom_training"] += 1
                self.timings["ml_rom_training"] += ml_rom_training_time
                self.detailed_results.append(("ml_rom_training", ml_rom_training_time))

            self.logger.info("Returning solution of the full model ...")
            # Return solution
            return u_fom
