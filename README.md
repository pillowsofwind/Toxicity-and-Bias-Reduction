# Toxicity-and-Bias-Reduction

This is the code repo for the EMNLP'24 🌴 paper: [Walking in Others' Shoes: How Perspective-Taking Guides Large Language Models in Reducing Toxicity and Bias](https://arxiv.org/pdf/2407.15366)

## Datasets: BOLD-1.5K and RTP-High

Please check out the two datasets under `.../datasets/data`.

## Code for Detoxification and Debiasing

In order to run the experiments, please first config your model in `.../utils.py`. For all other methods except for SHAP, you can first config `.../run_final.py` by setting the model and methods. For SHAP, refer to `.../shap_test.py`. Note that you should first run the base generations to use some of the methods/baselines.

## Code for Evaluation

For evaluation, please use `.../evaluates_all.py`, you can set the evaluation criteria in the main function manually.

## Cite Our Research

You may cite our research if you find it useful:
```
@misc{xu2024walkingothersshoesperspectivetaking,
      title={Walking in Others' Shoes: How Perspective-Taking Guides Large Language Models in Reducing Toxicity and Bias}, 
      author={Rongwu Xu and Zi'an Zhou and Tianwei Zhang and Zehan Qi and Su Yao and Ke Xu and Wei Xu and Han Qiu},
      year={2024},
      eprint={2407.15366},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.15366}, 
}
```
