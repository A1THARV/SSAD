"""Spectral-spatial anomaly detection prototype.

Implements the four-objective pipeline described in
*AI/ML Based Spectral-Spatial Anomaly Detection in Hyperspectral and Thermal
(IR) Remote Sensing Datasets*: hyperspectral anomaly detection, spectral
signature matching, thermal anomaly detection, and thermal emissivity-based
material identification.

Equation-to-code traceability (PRD Appendix A):

===== =========================================== =========================================
Eq.   Description                                 Location
===== =========================================== =========================================
1     Reconstruction error S(x) = ||x - x_hat||^2 ``hyperspectral.autoencoder``
2     RX detector (x-mu)^T Sigma^-1 (x-mu)        ``hyperspectral.rx_detector``
3     Composite training loss                     ``hyperspectral.autoencoder``
4     SAM theta(x,r) = acos(x.r / |x||r|)         ``hyperspectral.sam``
5     Planck's law L(lambda,T) = eps * B          ``thermal.planck_emissivity``
===== =========================================== =========================================
"""

__version__ = "0.1.0"
