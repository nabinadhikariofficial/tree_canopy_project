## Method
### Supervised Canopy Regression with Temporal Consistency

The proposed method predicts tree canopy cover from multi-channel annual raster patches using a supervised regression model augmented with temporal consistency regularization. Let \(x_t \in \mathbb{R}^{C \times H \times W}\) denote an input patch from year \(t\), where \(C\) is the number of input channels and \(H \times W\) is the spatial patch size. In the final pipeline, \(C=5\), corresponding to NLCD canopy, NLCD land cover, NDVI, land surface temperature, and PRISM mean temperature. The model outputs a dense canopy prediction \(\hat{y}_t\) for the same patch.

For years with canopy reference labels, the model is trained with a supervised regression loss. If \(y_t\) denotes the target canopy raster derived from the aligned lidar-based canopy reference, the supervised term is:

\[
L_{\text{sup}} = \mathrm{MSE}(\hat{y}_t, y_t)
\]

where mean squared error is computed only over valid labeled pixels.

To exploit the temporal structure of the dataset, a temporal consistency term is added between patches extracted from the same spatial location but different years. Let \(x_t\) and \(x_{t'}\) denote two co-registered patches from years \(t\) and \(t'\). Let \(f(\cdot)\) denote the model representation used for consistency regularization. In this project, \(f(\cdot)\) can be interpreted as the learned feature representation or the prediction-level response produced by the network for the input patch. The consistency loss is defined as:

\[
L_{\text{cons}} = \left\| f(x_t) - f(x_{t'}) \right\|_2^2
\]

This term encourages the model to produce temporally stable representations for the same geographic location across years, while still allowing the supervised term to fit the available canopy labels. The full training objective is:

\[
L = L_{\text{sup}} + \lambda L_{\text{cons}}
\]

where \(\lambda\) is a hyperparameter controlling the strength of the temporal regularization.

The motivation for this formulation is that canopy structure does not change arbitrarily from one year to the next at most locations. Therefore, even when dense labels are unavailable for intermediate years, temporally adjacent observations still provide weak supervisory information. The consistency term allows the model to use this structure to learn smoother and more robust representations over time.

In practice, the method was implemented with patch-based dense regression. Supervised patches were drawn from labeled years, while consistency pairs were formed from the same spatial coordinates across multiple years. The baseline model used only the supervised loss, whereas the consistency-enhanced model used the combined loss above. This makes temporal consistency the main methodological extension beyond standard supervised canopy regression.
