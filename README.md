# Experimental Code for Sawaya (2025).

This repository provides the experimental code to reproduce all figures and tables reported in the paper:

**Kazuma Sawaya, "Provable FDR Control for Deep Feature Selection".**

It contains implementations of the **proposed method** as well as **baseline methods** such as Neural Gaussian Mirror (NGM) and DeepLINK.

---

## 📁 Directory Structure

### `Proposed/`
Contains all experiment scripts related to the **proposed feature selection method**.

| File | Description |
|------|-------------|
| `asyN.py` | Generates the results for **Section 5.1**, evaluating asymptotic normality under the standard design. |
| `asyN_qq.py` | Produces the QQ-plots corresponding to **Section 5.1**. |
| `asyN_elpt.py` | Creates **Figure 4**, evaluating asymptotic normality under an **elliptical AR(1)** design. |
| `asyN_clsf.py` | Generates **Figure 8**, showing results for the **multi-class classification** setting. |
| `parallel_runner_origin.py` | Executes the main feature-selection experiments reported in **Section 5.2** (see usage below). |
| `parallel_runner_tr.py` | Same as above but dedicated to the **Transformer** model experiments. |
| `p_runner_origin_elpt.py` | Generates the **Appendix Figure 5** results (elliptical design). |
| `p_runner_clsf.py` | Generates the **Appendix Figure 10** results (multi-class classification). |

---

### `NGM/`
Implementation of the **Neural Gaussian Mirror (NGM)** baseline used for comparison.

| File | Description |
|------|-------------|
| `run_sngm_expts.py` | Main entry script to reproduce the NGM experiments. |

---

### `DeepLINK/`
Implementation of the **DeepLINK** baseline (a modified version of the official GitHub repository).

| File | Description |
|------|-------------|
| `run_deeplink_expts.py` | Main entry script to reproduce the DeepLINK experiments. |

---

## 🚀 Running the Main Experiment

The main experiment (Section 5.2) can be executed using `parallel_runner_origin.py`.

### Example Usage

```bash
# Launch all seeds in parallel (across available GPUs)
python parallel_runner_origin.py \
  --mode launch \
  --seeds 1-20 \
  --m 4000 --n 400 --T 3000 \
  --outdir res/iter

# Merge all seed results into a single directory
python parallel_runner_origin.py \
  --mode merge \
  --outdir res/iter
