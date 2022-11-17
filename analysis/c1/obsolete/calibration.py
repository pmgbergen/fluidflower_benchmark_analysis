# NOTE: This is unfinished business.


class _TailoredConcentrationAnalysis(daria.ConcentrationAnalysis):
    def __init__(self, base, segmentation, color, resize_factor, **kwargs) -> None:
        super().__init__(base, color, **kwargs)
        self.resize_factor = resize_factor

        # Initialize scaling parameters for all facies
        self.segmentation = segmentation
        self.number_segments = np.unique(segmentation).shape[0]
        assert np.isclose(np.unique(segmentation), np.arange(self.number_segments))
        self.scaling = np.ones(self.number_segments, dtype=float)

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        signal = cv2.resize(
            signal,
            None,
            fx=self.resize_factor,
            fy=self.resize_factor,
            interpolation=cv2.INTER_AREA,
        )
        signal = skimage.restoration.denoise_tv_chambolle(signal, 0.1)
        signal = np.atleast_3d(signal)
        return super().postprocess_signal(signal)

    def convert_signal(self, signal: np.ndarray) -> np.ndarray:
        return np.clip(
            np.multiply(self.scaling_array, processed_signal) + self.offset, 0, 1
        )

    def _determine_scaling(self) -> None:
        self.scaling_array = np.ones(self.base.shape[:2], dtype=float)
        for i in np.unique(self.segmentation):
            mask = self.segmentation == i
            self.scaling_array[mask] = self.scaling[i]

    def calibrate(
        self,
        injection_rate: float,
        images: list[daria.Image],
        initial_guess: Optional[list[tuple[float]]] = None,
        tol: float = 1e-3,
        maxiter: int = 20,
    ) -> None:
        """
        Calibrate the conversion used in __call__ such that the provided
        injection rate is matched for the given set of images.

        Args:
            injection_rate (float): constant injection rate in ml/hrs.
            images (list of daria.Image): images used for the calibration.
            initial_guess (list of tuple): intervals of scaling values to be considered
                in the calibration; need to define lower and upper bounds on
                the optimal scaling parameter.
            tol (float): tolerance for the bisection algorithm.
            maxiter (int): maximal number of bisection iterations used for
                calibration.
        """
        # TODO wrap in another loop?
        converged = np.zeros(self.num_segments, dtype=bool)
        visited = np.zeros(self.num_segments, dtype=bool)

        while False:

            # Visit each pos once
            for i in range(self.num_segments):

                # Find pos with largest sensitivity which
                sensitivity = -1
                for pos_candidate in np.where(~visited)[0]:
                    # TODO quantiy sensitivity
                    pos_sensisitivity = None
                    if pos_sensitivity > sensitivity:
                        pos = pos_candidate
                        sensitivity = pos_sensitivity

                # TODO need to perform minimization and not strict root problem.

                # Define a function which is zero when the conversion parameters are chosen properly.
                def deviation_sqr(scaling: float):
                    self.scaling[~visited] = scaling
                    return (injection_rate - self._estimate_rate(images)[0]) ** 2

                # Perform bisection
                self.scaling, success, _ = minimize(
                    deviation,
                    initial_guess[pos],
                    method="BFGS",
                    xtol=tol,
                    maxiter=maxiter,
                )

                # Mark position as visited
                visited[pos] = True
                converged[pos] = sucess

            if converged.all():
                break

        print(f"Calibration results in scaling factor {self.scaling}.")

    def _estimate_rate(self, images: list[daria.Image]) -> tuple[float]:
        """
        Estimate the injection rate for the given series of images.

        Args:
            images (list of daria.Image): basis for computing the injection rate.

        Returns:
            float: estimated injection rate.
            float: offset at time 0, useful to determine the actual start time,
                or plot the total concentration over time compared to the expected
                volumes.
        """
        # Conversion constants
        SECONDS_TO_HOURS = 1.0 / 3600.0
        M3_TO_ML = 1e6

        # Define reference time (not important which image serves as basis)
        ref_time = images[0].timestamp

        # For each image, compute the total concentration, based on the currently
        # set tuning parameters, and compute the relative time.
        total_volumes = []
        relative_times = []
        for img in images:

            # Fetch associated time for image, relate to reference time, and store.
            time = img.timestamp
            relative_time = (time - ref_time).total_seconds() * SECONDS_TO_HOURS
            relative_times.append(relative_time)

            # Convert signal image to concentration, compute the total volumetric
            # concentration in ml, and store.
            concentration = self(img)
            total_volume = self._determine_total_volume(concentration) * M3_TO_ML
            total_volumes.append(total_volume)

        # Determine slope in time by linear regression
        relative_times = np.array(relative_times).reshape(-1, 1)
        total_volumes = np.array(total_volumes)
        ransac = RANSACRegressor()
        ransac.fit(relative_times, total_volumes)

        # Extract the slope and convert to
        return ransac.estimator_.coef_[0], ransac.estimator_.intercept_
