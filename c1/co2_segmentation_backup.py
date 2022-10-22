## Define specific concentration analysis class to detect mobile CO2
#class MobileCO2Analysis(daria.BinaryConcentrationAnalysis):
#    def _extract_scalar_information(self, img: daria.Image) -> None:
#        pass
#
#    def _extract_scalar_information_after(self, img: np.ndarray) -> np.ndarray:
#        return signal[:, :, 0]
#
#
#class MobileCO2AnalysisPrior(daria.BinaryConcentrationAnalysis):
#
#    def _extract_scalar_information(self, img: daria.Image) -> None:
#        """Transform to HSV."""
#        img.img = cv2.cvtColor(img.img.astype(np.float32), cv2.COLOR_RGB2HSV)
#
#
#    def _extract_scalar_information_after(self, img: np.ndarray) -> np.ndarray:
#        """Return 3rd component of HSV (value), identifying signal strength."""
#        # Clip values in value - from calibration.
#        v_img = img[:,:,2]
#        # TODO include as parameters!
#        mask = v_img > 0.3
#
#        # Restrict to co2 mask
#        img[~mask] = 0
#
#        # Consider Value (3rd component from HSV).
#        return img[:,:,2]
#
#class MobileCO2AnalysisPosterior(daria.BinaryConcentrationAnalysis):
#
#    def _extract_scalar_information(self, img: daria.Image) -> None:
#        """Transform to HSV."""
#        img.img = cv2.cvtColor(img.img.astype(np.float32), cv2.COLOR_RGB2HSV)
#
#
#    def _extract_scalar_information_after(self, img: np.ndarray) -> np.ndarray:
#        """Return 3rd component of HSV (value), identifying signal strength."""
#        # Clip values in value - from calibration.
#        v_img = img[:,:,2]
#        # TODO include as parameters!
#        mask = v_img > 0.3
#
#        # Restrict to co2 mask
#        img[~mask] = 0
#
#        # Consider saturation(2nd component from HSV).
#        return img[:,:,1]

# ! ---- co2 mask - relevant

        ## Works for detecting all CO2
        #config_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 0.5,
        #    "presmoothing weight": 1,
        #    "presmoothing eps": 1e-4,
        #    "presmoothing max_num_iter": 1000,
        #    # Thresholding
        #    "threshold value": 0.04,
        #}

        ## Works for detecting all CO2
        #config_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 0.5,
        #    "presmoothing weight": 5,
        #    "presmoothing eps": 1e-4,
        #    "presmoothing max_num_iter": 1000,
        #    # Thresholding
        #    "threshold value": 0.04,
        #}

        #self.co2_mask_analysis = daria.BinaryConcentrationAnalysis(
        #    self.base_with_clean_water,
        #    color="red",
        #    **config_co2_analysis,
        #)

        #self._setup_concentration_analysis(
        #    self.co2_mask_analysis,
        #    "co2_mask_cleaning_filter_red.npy",
        #    baseline,
        #    update_setup,
        #)

        ## Value based, works well.
        ## Pros:
        ## Cons:
        #config_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 0.5,
        #    "presmoothing weight": 0.5,
        #    "presmoothing eps": 1e-4,
        #    "presmoothing max_num_iter": 100,
        #    # Thresholding
        #    "threshold value": 0.04,
        #}

        #self.co2_mask_analysis = daria.BinaryConcentrationAnalysis(
        #    self.base_with_clean_water,
        #    color="value",
        #    **config_co2_analysis,
        #)

        #self._setup_concentration_analysis(
        #    self.co2_mask_analysis,
        #    "co2_mask_cleaning_filter_value.npy",
        #    baseline,
        #    update_setup,
        #)


# ! ---- co2 mask - old

        #config_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 1,
        #    #"presmoothing weight": 100, # chambolle
        #    "presmoothing weight": 0.001,
        #    "presmoothing eps": 1e-4,
        #    "presmoothing max_num_iter": 100,
        #    "presmoothing method": "anisotropic bregman",
        #    # Thresholding
        #    "threshold value": 0.24,
        #    # Postsmoothing
        #    "postsmoothing": False,
        #}

        ## Red based filter for detecting the mobile CO2 phase.
        ## Also detects wrong CO2 regions. In addition, tiny regions are not
        ## correctly. Could potentially be used as prior.
        ## Anisotropic TVD is not too satisfying. It generates quite
        ## nonsmooth structures in fact.
        ## When using chambolle with weight 10, threshold value 0.22 seems slightly
        ## too low on the lower part, maybe OK for the upper part.
        #config_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 1,
        #    "presmoothing weight": 10, # chambolle
        #    #"presmoothing weight": 10,
        #    "presmoothing eps": 1e-6,
        #    "presmoothing max_num_iter": 1000,
        #    #"presmoothing method": "anisotropic bregman",
        #    # Thresholding
        #    "threshold value": 0.22,
        #    # Postsmoothing
        #    "postsmoothing": False,
        #}


        #self.co2_mask_analysis = daria.BinaryConcentrationAnalysis(
        #    self.base_with_clean_water,
        #    color="red",
        #    **config_co2_analysis,
        #)

        #self._setup_concentration_analysis(
        #    self.co2_mask_analysis,
        #    "co2_mask_cleaning_filter_red.npy",
        #    baseline,
        #    update_setup,
        #)

        ## Hue based thresholding - has worked well before
        #config_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    #"presmoothing resize": 1.,
        #    #"presmoothing weight": 10,
        #    #"presmoothing eps": 1e-5,
        #    #"presmoothing max_num_iter": 1000,
        #    #"presmoothing resize": 0.25,
        #    #"presmoothing weight": 1,
        #    #"presmoothing eps": 1e-5,
        #    #"presmoothing max_num_iter": 1000,
        #    #"presmoothing method": "chambolle",
        #    "presmoothing resize": 0.25,
        #    "presmoothing weight": 1,
        #    "presmoothing eps": 1e-5,
        #    "presmoothing max_num_iter": 1000,
        #    "presmoothing method": "anisotropic bregman",
        #    # Thresholding
        #    "threshold value": 10,
        #    # Remove small objects
        #    "min area size": 20**2,
        #    # Hole filling
        #    "max hole size": 20**2,
        #    # Local convex cover
        #    "local convex cover patch size": 10,
        #    # Postsmoothing
        #    "postsmoothing": True,
        #    #"postsmoothing resize": 0.25,
        #    #"postsmoothing weight": 10,
        #    #"postsmoothing eps": 1e-5,
        #    #"postsmoothing max_num_iter": 100,
        #    #"postsmoothing method": "chambolle",
        #    "presmoothing resize": 0.25,
        #    "presmoothing weight": 1,
        #    "presmoothing eps": 1e-5,
        #    "presmoothing max_num_iter": 1000,
        #    "presmoothing method": "anisotropic bregman",
        #}

        ## Does not work as nicely as before...
        #config_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 0.5,
        #    "presmoothing weight": 1,
        #    "presmoothing eps": 1e-4,
        #    "presmoothing max_num_iter": 100,
        #    # Small objects and holes
        #    "min area size": 20**2,
        #    "max hole size": 10**2,
        #    # Thresholding
        #    "threshold value": 5,
        #    # Presmoothing
        #    "postsmoothing": True,
        #    "postsmoothing resize": 0.5,
        #    "postsmoothing weight": 1,
        #    "postsmoothing eps": 1e-4,
        #    "postsmoothing max_num_iter": 100,
        #}

        #self.co2_mask_analysis = daria.BinaryConcentrationAnalysis(
        #    self.base_with_clean_water,
        #    color="hue",
        #    **config_co2_analysis,
        #)

        #self._setup_concentration_analysis(
        #    self.co2_mask_analysis,
        #    "co2_mask_cleaning_filter_hue.npy",
        #    baseline,
        #    update_setup,
        #)

        #config_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    #"presmoothing resize": 1.,
        #    #"presmoothing weight": 10,
        #    #"presmoothing eps": 1e-5,
        #    #"presmoothing max_num_iter": 1000,
        #    #"presmoothing resize": 0.25,
        #    #"presmoothing weight": 1,
        #    #"presmoothing eps": 1e-5,
        #    #"presmoothing max_num_iter": 1000,
        #    #"presmoothing method": "chambolle",
        #    "presmoothing resize": 0.25,
        #    "presmoothing weight": 1,
        #    "presmoothing eps": 1e-5,
        #    "presmoothing max_num_iter": 100,
        #    "presmoothing method": "anisotropic bregman",
        #    # Thresholding
        #    "threshold value": 10,
        #    # Remove small objects
        #    "min area size": 20**2,
        #    # Hole filling
        #    "max hole size": 20**2,
        #    # Local convex cover
        #    "local convex cover patch size": 10,
        #    # Postsmoothing
        #    "postsmoothing": True,
        #    #"postsmoothing resize": 0.25,
        #    #"postsmoothing weight": 10,
        #    #"postsmoothing eps": 1e-5,
        #    #"postsmoothing max_num_iter": 100,
        #    #"postsmoothing method": "chambolle",
        #    "presmoothing resize": 0.25,
        #    "presmoothing weight": 1,
        #    "presmoothing eps": 1e-5,
        #    "presmoothing max_num_iter": 100,
        #    "presmoothing method": "anisotropic bregman",
        #}

# ! ---- mobile co2

        ## Red based - works OKish for some time steps but not nice at the end.
        # config_mobile_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 0.5,
        #    "presmoothing weight": 5,
        #    "presmoothing eps": 1e-4,
        #    "presmoothing max_num_iter": 100,
        #    # Thresholding
        #    "threshold value": 0.24,
        #    ## Presmoothing
        #    #"postsmoothing": True,
        #    #"postsmoothing resize": 0.5,
        #    #"postsmoothing weight": 5,
        #    #"postsmoothing eps": 1e-4,
        #    #"postsmoothing max_num_iter": 100,
        # }

        # self.mobile_co2_analysis = daria.BinaryConcentrationAnalysis(
        #    self.base_with_clean_water,
        #    color="red",
        #    **config_mobile_co2_analysis,
        # )

        # self._setup_concentration_analysis(
        #    self.mobile_co2_analysis,
        #    "co2_mask_cleaning_filter_red.npy",
        #    baseline,
        #    update_setup,
        # )

        ## RGB on diff, restrict to blue afterwards
        #config_mobile_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 0.5,
        #    "presmoothing weight": 5,
        #    "presmoothing eps": 1e-4,
        #    "presmoothing max_num_iter": 100,
        #    # Thresholding
        #    "threshold value": 0.04,
        #    ## Presmoothing
        #    #"postsmoothing": True,
        #    #"postsmoothing resize": 0.5,
        #    #"postsmoothing weight": 5,
        #    #"postsmoothing eps": 1e-4,
        #    #"postsmoothing max_num_iter": 100,
        #}

        #self.mobile_co2_analysis = MobileCO2Analysis(
        #    self.base_with_clean_water,
        #    color="",
        #    **config_mobile_co2_analysis,
        #)

        #self._setup_concentration_analysis(
        #    self.mobile_co2_analysis,
        #    "co2_mask_cleaning_filter_vector_blue.npy",
        #    baseline,
        #    update_setup,
        #)

        ## Concentration analysis to detect mobile CO2. Hue serves as basis for the analysis.
        #config_mobile_co2_analysis = {
        #   # Presmoothing
        #   "presmoothing": True,
        #   "presmoothing resize": 0.5,
        #   "presmoothing weight": 1,
        #   "presmoothing eps": 1e-4,
        #   "presmoothing max_num_iter": 100,
        #   "presmoothing method": "chambolle",

        #   # Thresholding
        #   "threshold value": 0.05,

        #   ## Remove small objects
        #   #"min area size": 50**2, # TODO 

        #   ## Hole filling
        #   #"max hole size": 1,

        #   ## Local convex cover
        #   #"local convex cover patch size": 1,

        #   # Postsmoothing
        #   "postsmoothing": False,
        #}

        #self.mobile_co2_analysis = MobileCO2AnalysisPrior(
        #   self.base_with_clean_water,
        #   color="",
        #   **config_mobile_co2_analysis
        #)

        #print("Warning: Concentration analysis is not calibrated.")

        #self._setup_concentration_analysis(
        #    self.mobile_co2_analysis,
        #    "mobile_co2_prior_cleaning_filter.npy",
        #    baseline,
        #    update_setup,
        #)

        ## Concentration analysis to detect mobile CO2. Hue serves as basis for the analysis.
        #config_mobile_co2_analysis_posterior = {
        #   # Presmoothing
        #   "presmoothing": True,
        #   "presmoothing resize": 0.5,
        #   "presmoothing weight": 1,
        #   "presmoothing eps": 1e-4,
        #   "presmoothing max_num_iter": 100,
        #   "presmoothing method": "chambolle",

        #   # Thresholding
        #   "threshold value": 0.03,

        #   # Remove small objects
        #   "min area size": 50**2, # TODO 

        #   # Hole filling
        #   "max hole size": 1,

        #   # Local convex cover
        #   "local convex cover patch size": 1,

        #   # Postsmoothing
        #   "postsmoothing": False,
        #}

        #self.mobile_co2_analysis_posterior = MobileCO2Analysis(
        #   self.base_with_clean_water,
        #   color="",
        #   **config_mobile_co2_analysis_posterior
        #)

        #print("Warning: Concentration analysis is not calibrated.")

        #self._setup_concentration_analysis(
        #    self.mobile_co2_analysis_posterior,
        #    "mobile_co2_posterior_cleaning_filter.npy",
        #    baseline,
        #    update_setup,
        #)
       
        ## Concentration analysis to detect mobile CO2. Hue serves as basis for the analysis.
        #config_mobile_co2_analysis = {
        #   # Presmoothing
        #   "presmoothing": True,
        #   "presmoothing resize": 1.,
        #   "presmoothing weight": 1,
        #   "presmoothing eps": 1e-5,
        #   "presmoothing max_num_iter": 100,
        #   "presmoothing method": "chambolle",

        #   # Thresholding
        #   "threshold value": 0.048, # for blue
        #   #"threshold value": 0.25, # for red

        #   # Hole filling
        #   "max hole size": 20**2,

        #   # Local convex cover
        #   "local convex cover patch size": 10,

        #   # Postsmoothing
        #   "postsmoothing": True,
        #   "postsmoothing resize": 0.25,
        #   "postsmoothing weight": 4, # 4 if resize=0.25
        #   "postsmoothing eps": 1e-5,
        #   "postsmoothing max_num_iter": 100,
        #   "postsmoothing method": "chambolle"
        #}
       
        #self.mobile_co2_analysis = MobileCO2Analysis(
        #   self.base_with_clean_water,
        #   color="empty",
        #   **config_mobile_co2_analysis
        #)

        #print("Warning: Concentration analysis is not calibrated.")

        #self._setup_concentration_analysis(
        #    self.mobile_co2_analysis,
        #    "mobile_co2_cleaning_filter.npy",
        #    baseline,
        #    update_setup,
        #)


        ## Concentration analysis to detect mobile CO2. Hue serves as basis for the analysis.
        #config_mobile_co2_analysis = {
        #    # Presmoothing
        #    "presmoothing": True,
        #    "presmoothing resize": 0.5,
        #    "presmoothing weight": 0.01,
        #    "presmoothing eps": 1e-5,
        #    "presmoothing max_num_iter": 1000,
        #    "presmoothing method": "anisotropic bregman",
        #    # Thresholding
        #    "threshold value": 20,
        #    # Hole filling
        #    "max hole size": 20**2,
        #    # Local convex cover
        #    "local convex cover patch size": 10,
        #    # Postsmoothing
        #    "postsmoothing": False,
        #    # "postsmoothing resize": 0.25,
        #    # "postsmoothing weight": 4, # 4 if resize=0.25
        #    # "postsmoothing eps": 1e-5,
        #    # "postsmoothing max_num_iter": 100,
        #    # "postsmoothing method": "chambolle"
        #}

        #self.mobile_co2_analysis = MobileCO2AnalysisNew(
        #    self.base_with_clean_water, color="empty", **config_mobile_co2_analysis
        #)

        #print("Warning: Concentration analysis is not calibrated.")

        #self._setup_concentration_analysis(
        #    self.mobile_co2_analysis,
        #    "mobile_co2_cleaning_filter_new.npy",
        #    baseline,
        #    update_setup,
        #)

# ! ---- prior / posterior case


#        # Extract concentration map and gradient of smooth signal
#        mobile_co2_prior = self.mobile_co2_analysis(img)
#        grad_mod = self.mobile_co2_analysis.gradient_modulus()
#
#        # ! ---- posterior
#
#        ## Mark co2 as active set, but turn off esf
#        #self.mobile_co2_analysis_posterior.update_mask(
#        #    np.logical_and(
#        #        co2.img,
#        #        np.logical_not(self.esf),
#        #    )
#        #)
#
#        # Extract concentration map
#        mobile_co2_posterior = np.zeros(img.img.shape, dtype=bool)
#
#        # Label the connected regions first
#        labels_prior, num_labels_prior = skimage.measure.label(mobile_co2_prior.img, return_num=True)
#        props = skimage.measure.regionprops(labels_prior)
#
#        # Investigate each labeled region separately; omit label 0, which corresponds to non-marked area.
#        for label in range(1,num_labels_prior+1):
#
#            # Fix one label
#            labeled_region = labels_prior == label
#
#            # Determine contour set of labeled region
#            contours, _ = cv2.findContours(
#                skimage.util.img_as_ubyte(labeled_region), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
#            )
#
#            # For each part of the contour set, check whether the gradient is sufficiently
#            # large at any location
#            accept = False
#            for c in contours:
#
#                # Extract coordinates of contours
#                c = np.fliplr(np.squeeze(c))
#                c = (c[:,0], c[:,1])
#
#                # Identify region as marked if gradient sufficiently large
#                if np.max(grad[c]) >  0.002: #self.threshold_gradient: # TODO include as parameter
#                    accept = True
#                    break
#
#            # Collect findings
#            if accept:
#                mobile_co2_posterior[labeled_region] = True
#
#        # Overlay prior and posterior
#        mobile_co2 = np.zeros(img.img.shape, dtype=bool)
#        for label in range(1,num_labels_prior+1):
#
#            # Fix one label
#            labeled_region = labels_prior == label
#
#            # Check whether posterior marked in this area
#            if np.any(mobile_co2_posterior[labeled_region]):
#                mobile_co2[labeled_region] = True

# ! ---- from determine mobile co2 mask

#        # Extract concentration map and gradient of smooth signal
#        mobile_co2_prior = self.mobile_co2_analysis(img)
#        grad_mod = self.mobile_co2_analysis.gradient_modulus()
#
#        # ! ---- posterior
#
#        ## Mark co2 as active set, but turn off esf
#        #self.mobile_co2_analysis_posterior.update_mask(
#        #    np.logical_and(
#        #        co2.img,
#        #        np.logical_not(self.esf),
#        #    )
#        #)
#
#        # Extract concentration map
#        mobile_co2_posterior = np.zeros(img.img.shape, dtype=bool)
#
#        # Label the connected regions first
#        labels_prior, num_labels_prior = skimage.measure.label(mobile_co2_prior.img, return_num=True)
#        props = skimage.measure.regionprops(labels_prior)
#
#        # Investigate each labeled region separately; omit label 0, which corresponds to non-marked area.
#        for label in range(1,num_labels_prior+1):
#
#            # Fix one label
#            labeled_region = labels_prior == label
#
#            # Determine contour set of labeled region
#            contours, _ = cv2.findContours(
#                skimage.util.img_as_ubyte(labeled_region), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
#            )
#
#            # For each part of the contour set, check whether the gradient is sufficiently
#            # large at any location
#            accept = False
#            for c in contours:
#
#                # Extract coordinates of contours
#                c = np.fliplr(np.squeeze(c))
#                c = (c[:,0], c[:,1])
#
#                # Identify region as marked if gradient sufficiently large
#                if np.max(grad[c]) >  0.002: #self.threshold_gradient: # TODO include as parameter
#                    accept = True
#                    break
#
#            # Collect findings
#            if accept:
#                mobile_co2_posterior[labeled_region] = True
#
#        # Overlay prior and posterior
#        mobile_co2 = np.zeros(img.img.shape, dtype=bool)
#        for label in range(1,num_labels_prior+1):
#
#            # Fix one label
#            labeled_region = labels_prior == label
#
#            # Check whether posterior marked in this area
#            if np.any(mobile_co2_posterior[labeled_region]):
#                mobile_co2[labeled_region] = True

