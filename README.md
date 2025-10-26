<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# CUSTOMER-CHURN-TENSORFLOW

<em></em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/LordZeusIsBack/Customer-Churn-TensorFlow?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/LordZeusIsBack/Customer-Churn-TensorFlow?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/LordZeusIsBack/Customer-Churn-TensorFlow?style=default&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/LordZeusIsBack/Customer-Churn-TensorFlow?style=default&color=0080ff" alt="repo-language-count">

<!-- default option, no dependency badges. -->

<!-- default option, no dependency badges. -->

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
  - [Project Index](#project-index)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Customer-Churn-TensorFlow is an end-to-end churn prediction workspace featuring Keras-based models trained on customer attributes and behavior signals, with saved artifacts for consistent inference across environments. The repo contains training and inference notebooks, a runnable app.py entry point, fitted preprocessing assets (scaler and geography encoder), and baseline regression/classification variants to benchmark results. After installing requirements, you can run the app for interactive scoring or use the prediction notebook to batch-score datasets with the provided model and preprocessors.

---

## Features

Binary churn prediction using a TensorFlow Keras model trained on customer demographics, account metrics, and geography, with saved artifacts for reproducible inference.

---

## Project Structure

```sh
└── Customer-Churn-TensorFlow/
    ├── LICENSE
    ├── app.py
    ├── experiments.ipynb
    ├── geo_encoder.pkl
    ├── model.keras
    ├── predictions.ipynb
    ├── regression inference.ipynb
    ├── regression.ipynb
    ├── regression.keras
    ├── regression.pkl
    ├── requirements.txt
    └── scaler.pkl
```

### Project Index

<details open>
	<summary><b><code>CUSTOMER-CHURN-TENSORFLOW/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/LICENSE'>LICENSE</a></b></td>
					<td style='padding: 8px;'>Repository license for Customer-Churn-TensorFlow.</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/app.py'>app.py</a></b></td>
					<td style='padding: 8px;'>Streamlit entry script for loading scalers/encoders and serving churn predictions from saved Keras and pickle artifacts.</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/experiments.ipynb'>experiments.ipynb</a></b></td>
					<td style='padding: 8px;'>Exploratory experiments for feature engineering, model trials, and evaluation.</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/model.keras'>model.keras</a></b></td>
					<td style='padding: 8px;'>Trained TensorFlow Keras classification model for churn prediction.</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/predictions.ipynb'>predictions.ipynb</a></b></td>
					<td style='padding: 8px;'>Notebook for batch/interactive predictions using saved model and preprocessors.</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/regression inference.ipynb'>regression inference.ipynb</a></b></td>
					<td style='padding: 8px;'>Inference-only workflow for regression baseline variant on churn-related target.</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/regression.ipynb'>regression.ipynb</a></b></td>
					<td style='padding: 8px;'>Training notebook for regression baseline and comparisons with classification model.</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/regression.keras'>regression.keras</a></b></td>
					<td style='padding: 8px;'>Saved Keras model for the regression baseline.</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>Python dependencies to reproduce the environment.</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** JupyterNotebook
- **Package Manager:** Pip

### Installation

Build Customer-Churn-TensorFlow from the source and intsall dependencies:

1. **Clone the repository:**

   ```sh
   ❯ git clone https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow.git
   ```

2. **Navigate to the project directory:**

   ```sh
   ❯ cd Customer-Churn-TensorFlow
   ```

3. **Install the dependencies:**

   ```sh
   ❯ pip install -r requirements.txt
   ```

### Usage

Run the project with:

```sh
❯ streamlit run app.py
```

---

## Contributing

- **💬 [Join the Discussions](https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/issues)**: Submit bugs found or log feature requests for the `Customer-Churn-TensorFlow` project.
- **💡 [Submit Pull Requests](https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/LordZeusIsBack/Customer-Churn-TensorFlow/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=LordZeusIsBack/Customer-Churn-TensorFlow">
   </a>
</p>
</details>

---

## License

Customer-churn-tensorflow is protected under the [LICENSE](https://mit-license.org/) License. For more details, refer to the [LICENSE](https://github.com/LordZeusIsBack/Customer-Churn-TensorFlow/blob/main/LICENSE) file.

---

<div align="right">

[![][back-to-top]](#top)

</div>

[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square
