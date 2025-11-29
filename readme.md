# XRayLogSurrogate
**Real-Time X-ray Beamline Surrogate Modeling via the First Physics-Informed Log-Manifold Learning Framework**
<!---
[![IPAC'26 Contribution](https://img.shields.io/badge/IPAC'26-Contribution-blue)](https://www.ipac26.org/)
--->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Affiliation](https://img.shields.io/badge/Tsinghua-SSMB-purple)](https://www.tsinghua.edu.cn/en/)

<!---
> **Supplementary Material for IPAC'26 Contribution**  
> *"End-to-End Surrogate Modeling of Undulator Radiation Transport in X-ray Beamlines: Accurately Predicting High-Dynamic-Range Focal Spots via Log-Manifold Learning"*
--->
---

##  Abstract

**XRayLogSurrogate** is the first physics-informed surrogate modeling framework capable of reproducing entire wave-optical propagation of undulator radiation through complex X-ray beamlines with **sub-percent accuracy** across extreme **dynamic range** of focal spot distributions in intensity.

- **The Challenge:** Wave-optical simulation of undulator radiation through entire beamlines is computationally prohibitive, hindering real-time parameter optimization.The high-frequency diffraction structures and extreme dynamic range of focal spot distributions pose significant challenges to conventional surrogate models.
- **The Innovation:** We propose the **first log-manifold surrogate** modeling framework that represents intensity distributions in logarithmic space, converting highly nonlinear diffraction structures into low-rank learnable manifolds.Combined with **physics-informed constraints**, the model achieves high accuracy with minimal training samples and generalizes well across beamline configurations.
- **The Result:** The proposed model achieves <1% relative error across the full dynamic range, faithfully reconstructing fine diffraction fringes. Single prediction takes only milliseconds, yielding a **$>10^4$ speedup** over SRW simulation and enabling real-time surrogate-based beamline optimization.

---

##  Key Performance Metrics

| Metric | **Log-Manifold Model (Ours)** | Linear-Space Baseline |
| :--- | :--- | :--- |
| **Mean Error** | **< 1 %** | > 100 % |
| **Simulation Time**  | **1.8 ms** | 2.1 ms |
| **Speedup vs. SRW** | **> 10,000×** | — |

---

##  Methodology: Log-Manifold Learning

### 1. The Logarithmic Mapping
Standard neural networks struggle with the extreme dynamic range of X-ray focal spots. We convert the intensity field $I(x,y)$ into a logarithmic manifold:

$$
L(x,y) = \log_{10}(I(x,y) + \epsilon), \quad  \epsilon\ll 1
$$

This logarithmic mapping dramatically linearizes the otherwise severely nonlinear diffraction fringes, yielding a **low-rank, highly learnable** manifold in log-space (see visualization below).

<p align="center">
  <img src="images/linear_and_log.png" width="800" alt="Linear vs Log Representation">
  <br>
  <em><b>Figure 1:</b> Intensity focal spot in linear (left) and log₁₀ (right) scale. The logarithmic representation dramatically improves the learnability of fringes.</em>
</p>

### 2. Log-Manifold POD

To exploit the dramatically improved linearity revealed by the logarithmic mapping, we perform Proper Orthogonal Decomposition (POD) **exclusively in the log₁₀-intensity domain**.  

Let $\{L_n(x,y)\}_{n=1}^{N_{\text{sample}}}$ denote the set of $N_{\text{sample}}$ centered log-intensity snapshots. The POD yields an optimal (in $L^2$ sense) orthonormal basis $\{\phi_k(x,y)\}_{k=1}^{r}$ and corresponding coefficients $\{a_{n,k}\}_{n,k}$ such that

$$
L_n(x,y) \approx \bar{L}(x,y) + \sum_{k=1}^{r} a_{n,k}\,\phi_k(x,y),
\quad r\ll N_{\text{pixel}},
$$

where $\bar{L}(x,y)$ is the ensemble mean and $r$ is the retained rank.

In the present case, **>99.99 % of the log-space energy is captured with only 58 modes** (versus more than 200 modes required in linear intensity space), confirming that the log-manifold is intrinsically low-rank.

<p align="center">
  <img src="images/log_pod_modes.png" width="820"/>
  <br>
  <b>Figure 2:</b> First sixteen POD modes  computed in logarithmic space. The modes exhibit clear, physically interpretable diffraction and caustic structures, in stark contrast to the noisy, high-rank modes obtained in linear intensity space .
</p>

<p align="center">
  <img src="images/eigen_value_convergence.png" width="420"/>
  <img src="images/linear_pod_modes.png" width="420"/>
  <br>
  <b>Figure 3:</b> (Left) Singular-value spectrum decay is ~3× faster in log-space. (Right) Corresponding first sixteen POD modes in linear intensity space for comparison.
</p>

### 3. Physics-Informed Residual Surrogate Network

The POD coefficients $a_k$ are predicted by a compact yet highly expressive **physics-informed residual neural network** that directly ingests upstream beamline parameters (undulator deflection parameter $K$, electron-beam emittance $\epsilon_{x,y}$, energy spread, mirror figure errors, etc.).

Key architectural and training enhancements that dramatically boost generalization, robustness, and physical fidelity:

- **Deep residual blocks** (8 layers) with SiLU activation and dropout for stable training of highly nonlinear mappings  
- **Energy-weighted coefficient loss** that prioritizes physically dominant low-order modes while preventing neglect of high-frequency diffraction details  
- **Explicit $L^2$-normalization constraint** on predicted coefficients to enforce conservation of log-space energy  
- **Physics-informed regularization terms** (undulator resonance condition + upstream wavefront curvature consistency) injected during training  
- **Input noise injection** on beam parameters to emulate realistic machine jitter and further enhance out-of-distribution generalization  

The combination of these ingredients yields **sub-percent relative error even on completely unseen beamline configurations**, while maintaining millisecond-level inference—more than four orders of magnitude faster than full SRW propagation.

<p align="center">
  <img src="images/framework.png" width="920" alt="Physics-informed residual surrogate"/>
  <br>
  <b>Figure 3:</b> End-to-end physics-informed residual surrogate architecture. 
</p>

---

## Validation & Reproducibility

All quantitative results reported in this work are obtained on a **fully open benchmark dataset** consisting of 200 high-fidelity SRW simulations of a realistic fourth-generation storage-ring beamline, with the undulator deflection parameter $K$ (or vertical magnetic field $B_y$) as the sole varying parameter.

- The log-manifold surrogate achieves **0.42 % mean relative $L^2$ error** over the entire focal plane and faithfully recovers diffraction fringes down to the $10^{-10}$ intensity level.
- In stark contrast, an identical architecture trained in linear intensity space completely fails to reproduce high-frequency structures .

<p align="center">
  <img src="images/log_predict_and_error.png" width="920"/>
  <br>
  <b>Figure 4:</b> Typical focal-spot reconstruction using the log-manifold surrogate (top) compared with ground-truth SRW simulation (middle) and point-wise relative error (bottom). 
</p>

<p align="center">
  <img src="images/reconstruction_prediction_varying_error.png" width="680"/>
  <br>
  <b>Figure 5:</b> Mean relative error versus varying parameter on 200 unseen test configurations .
</p>

## Full reproducibility 

The complete dataset, trained model weights, training/validation splits, and one-click visualization script are publicly available in the **demo/** folder of this repository under the MIT license, enabling immediate verification and extension by the community.

---

## Usage

To run the demo surrogate model, you first need to set up the base SRW environment, then install our specific dependencies.

### 1. Setup SRW Environment
This project relies on the **Synchrotron Radiation Workshop (SRW)** as the physics engine/ground truth generator.

1.  Clone the official SRW repository:
    ```bash
    git clone https://github.com/ochubar/SRW.git
    cd SRW
    ```
2.  Follow the installation instructions in the [SRW README](https://github.com/ochubar/SRW) to compile the C++ core and configure the Python interface (`srwpy`).
    *   *Note: Ensure that `srwpy` is accessible in your `PYTHONPATH`.*

### 2. Install Project Dependencies
Once SRW is configured, clone this repository and install the required Python packages:

```bash
git clone https://github.com/THUstarry/XRayLogSurrogate.git
cd XRayLogSurrogate
pip install -r requirements.txt
```

### 3. Run the Demo
We provide a demo dataset and a python script (`demo_model.py`) that loads the model definition & training & visualiztion code for the focal-plane intensity prediction for a sample beamline configuration.

```bash
python demo_model.py
```

**Example Output:**
The script will generate plot comparing the predicted focal spot against the ground truth.

---

##  Authors & Affiliation

- **First Author:** X. Song (songxy23@mail.tsinghua.edu.cn)
- **Supervisor** C.Tang (tang.xuh@tsinghua.edu.cn)
- **Corresponding Author:** J. S. (sjy22@mail.tsinghua.edu.cn)
- **co Auther:** L.H. (hul22@mails.tsinghua.edu.cn)
- **Affiliation:** Department of Engineering Physics, Tsinghua University

*This work contributes to advancing real-time digital-twin beamline modeling in the SSMB (Steady-State Microbunching) program in Tsinghua University.*

---
<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Validated%20against-SRW-006699?style=for-the-badge">
</p>