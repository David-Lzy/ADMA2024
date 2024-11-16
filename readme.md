<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">Correlation Analysis of Adversarial Attack on Time Series Classification</h1></p>
<p align="center">
 <em>Adma 2024 accepted paper.</em>
</p>

<p align="center">
 <img src="https://img.shields.io/github/license/David-Lzy/ADMA2024?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
 <img src="https://img.shields.io/github/last-commit/David-Lzy/ADMA2024?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
 <img src="https://img.shields.io/github/languages/top/David-Lzy/ADMA2024?style=default&color=0080ff" alt="repo-top-language">
 <img src="https://img.shields.io/github/languages/count/David-Lzy/ADMA2024?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
 <!-- default option, no dependency badges. -->
</p>
<br>

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
  - [Project Index](#-project-index)
- [Getting Started](#-getting-started)
  - [Prerequisites](#-prerequisites)
  - [Installation](#-installation)
  - [Usage](#-usage)
  - [Testing](#-testing)
- [Project Roadmap](#-project-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## Overview

ADMA2024 revolutionizes model robustness by integrating Gaussian smoothing and noise augmentation into machine learning pipelines. It dynamically manages memory constraints, ensuring efficient training and evaluation across datasets. Ideal for data scientists and ML engineers, it enhances model performance against adversarial conditions, fostering resilient AI solutions.

---

## Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Microservices-based architecture</li><li>Event-driven communication</li><li>Scalable and resilient design</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Adheres to PEP 8 standards</li><li>Extensive use of linters and formatters</li><li>High code coverage with unit tests</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive API documentation</li><li>Detailed setup and installation guides</li><li>Regularly updated changelog</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Seamless integration with Python libraries</li><li>Supports various data formats</li><li>Compatible with cloud services</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Highly modular codebase</li><li>Easy to extend and customize</li><li>Clear separation of concerns</li></ul> |
| 🧪 | **Testing**       | <ul><li>Automated testing pipeline</li><li>Includes unit, integration, and system tests</li><li>Continuous integration with test coverage reports</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimized for high throughput</li><li>Low latency processing</li><li>Efficient resource utilization</li></ul> |
| 🛡️ | **Security**      | <ul><li>Implements OAuth2 for authentication</li><li>Regular security audits</li><li>Data encryption in transit and at rest</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Utilizes a wide range of datasets</li><li>Dependencies managed via a virtual environment</li><li>Regular updates to dependencies</li></ul> |
| 🚀 | **Scalability**   | <ul><li>Horizontally scalable</li><li>Supports load balancing</li><li>Elastic scaling based on demand</li></ul> |

```

---

##  Project Structure

```sh

└── WorkSpace
    ├── CODE/
    │   ├── Attack
    │   │   ├── __pycache__
    │   │   ├── attacker.py
    │   │   ├── correlation.py
    │   │   ├── cosine.py
    │   │   ├── deepfool.py
    │   │   ├── fft.py
    │   │   ├── mix.py
    │   │   ├── pgd.py
    │   │   ├── swap.py
    │   │   └── swap_l2.py
    │   ├── Config
    │   │   ├── DEFAULT_ATTACK.json
    │   │   ├── DEFAULT_DATA_NAME.json
    │   │   ├── DEFAULT_TRAIN.json
    │   │   ├── full_pramater_attack.jsonc
    │   │   └── full_pramater_train.jsonc
    │   ├── Demo
    │   │   ├── ATTACK_no_defence
    │   │   ├── ATTACK_with_arg
    │   │   ├── Archived
    │   │   ├── TEST
    │   │   ├── train_noise.1.py
    │   │   ├── train_noise.2.py
    │   │   ├── train_noise.py
    │   │   ├── train_smooth.1.py
    │   │   ├── train_smooth.2.py
    │   │   ├── train_smooth.3.py
    │   │   └── train_smooth.4.py
    │   ├── Train
    │   │   ├── TS_2_V
    │   │   ├── defence.py
    │   │   ├── inception_time.py
    │   │   ├── lstm_fcn.py
    │   │   ├── macnn.py
    │   │   ├── resnet.py
    │   │   ├── trainer.py
    │   │   └── transformer.py
    │   ├── Utils
    │   │   ├── augmentation.py
    │   │   ├── constant.py
    │   │   ├── correlation.py
    │   │   ├── package.py
    │   │   └── utils.py
    │   ├── __init__.py
    │   └── ENV
    │       └── freeze.yml
    ├── DATA  # UCR2018 Dataset
    ├── LOG
    └── OUTPUT
```

 Note: [UCR2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) is here: <https://www.cs.ucr.edu/~eamonn/time_series_data_2018/>.

### Project Index

<details open>
 <summary><b><code>ADMA2024/</code></b></summary>
 <details> <!-- __root__ Submodule -->
  <summary><b>__root__</b></summary>
  <blockquote>
   <table>
   </table>
  </blockquote>
 </details>
 <details> <!-- Demo Submodule -->
  <summary><b>Demo</b></summary>
  <blockquote>
   <table>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/train_smooth.2.py'>train_smooth.2.py</a></b></td>
    <td>- Facilitates the training and evaluation of defense models using Gaussian smoothing as an augmentation method within a machine learning pipeline<br>- It iterates over datasets and models, adjusting batch sizes to manage memory constraints, and integrates with the broader architecture by leveraging shared utilities for augmentation and defense<br>- The process aims to enhance model robustness and performance through systematic training and evaluation.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/train_smooth.3.py'>train_smooth.3.py</a></b></td>
    <td>- Facilitates the training and evaluation of defense models on univariate datasets using Gaussian smoothing as an augmentation method<br>- Iterates through datasets and models, adjusting batch sizes to manage memory constraints, and employs a trainer to execute training and evaluation processes<br>- Integrates results by concatenating training and testing metrics, contributing to the project's focus on robust model training and evaluation.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/train_noise.1.py'>train_noise.1.py</a></b></td>
    <td>- Facilitates the training and evaluation of defense models against noise in univariate datasets by leveraging Gaussian noise augmentation<br>- Iterates through specified models and datasets, adjusting batch sizes dynamically to handle memory constraints<br>- Integrates with the broader architecture by utilizing augmentation and defense modules, ensuring robust model training and evaluation while managing computational resources effectively.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/train_smooth.1.py'>train_smooth.1.py</a></b></td>
    <td>- Facilitates the training and evaluation of defense models using Gaussian smoothing as an augmentation method on univariate datasets<br>- Iterates through various models and datasets, adjusting batch sizes to manage memory constraints, and employs a trainer to execute the training process on a CUDA device<br>- Aggregates training and testing metrics to assess model performance within the broader project architecture.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/train_noise.py'>train_noise.py</a></b></td>
    <td>- Facilitates the training and evaluation of machine learning models with noise augmentation, specifically using Gaussian noise, to enhance model robustness<br>- Iterates over predefined models and datasets, dynamically adjusting batch sizes to manage GPU memory constraints<br>- Integrates with the broader architecture by leveraging augmentation and defense mechanisms, ultimately contributing to the project's goal of improving model performance and reliability through systematic noise-based training.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/train_smooth.4.py'>train_smooth.4.py</a></b></td>
    <td>- Facilitates the training and evaluation of defense models using Gaussian smoothing augmentation on univariate datasets<br>- Iterates through datasets and models, adjusting batch sizes to manage CUDA memory constraints<br>- Integrates augmentation methods and model parameters to enhance model robustness<br>- Outputs training and testing metrics, contributing to the project's goal of improving model performance and resilience against adversarial attacks.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/train_noise.2.py'>train_noise.2.py</a></b></td>
    <td>- Facilitates the training and evaluation of defense models using Gaussian noise augmentation on univariate datasets<br>- Iterates through specified models and datasets, adjusting batch sizes to manage memory constraints<br>- Utilizes a trainer to execute training and evaluation processes, ensuring results are stored and metrics are concatenated for both training and testing phases, contributing to the project's robustness against noisy data.</td>
   </tr>
   </table>
   <details>
    <summary><b>ATTACK_no_defence</b></summary>
    <blockquote>
     <table>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_SWAP_epoch1000.py'>Attack_SWAP_epoch1000.py</a></b></td>
      <td>- Facilitates adversarial attacks on machine learning models by utilizing the SWAP attack method across various datasets and model architectures<br>- It systematically adjusts batch sizes to manage memory constraints and iteratively applies perturbations to evaluate model robustness<br>- The process integrates with the broader codebase by leveraging existing model definitions and training utilities, ultimately contributing to the project's focus on enhancing model security and resilience.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_SWAP.py'>Attack_SWAP.py</a></b></td>
      <td>- Facilitates an attack simulation using the SWAP method within the broader project framework<br>- It dynamically adjusts the system path to integrate necessary modules and executes the attack process with specified parameters, leveraging GPU resources<br>- This component is crucial for testing the system's resilience against specific attack vectors, contributing to the project's overall focus on security and robustness evaluation.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_PGD.1.py'>Attack_PGD.1.py</a></b></td>
      <td>- Facilitates adversarial attacks using the Projected Gradient Descent (PGD) method within the broader codebase architecture<br>- It dynamically adjusts the system path to import necessary modules and executes the attack process on a specified device, such as a GPU<br>- The script is designed to handle interruptions and runtime errors, ensuring robust execution of the attack mechanism in a machine learning context.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_PGD.py'>Attack_PGD.py</a></b></td>
      <td>- Facilitates adversarial attacks using the Projected Gradient Descent (PGD) method within a machine learning framework<br>- Integrates the PGD attack into the broader system by dynamically adjusting system paths and configurations<br>- Aims to test model robustness by executing attacks on models without defense mechanisms, leveraging GPU resources for computation<br>- Enhances the project's capability to evaluate and improve model security against adversarial threats.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_COS.py'>Attack_COS.py</a></b></td>
      <td>- Facilitates an attack simulation using the cosine similarity method within a machine learning context<br>- Integrates with the broader codebase by dynamically adjusting the system path to access necessary modules<br>- Configures specific parameters for the attack, such as the number of epochs and a constant value, and executes the attack on a CUDA-enabled device, ensuring adaptability to runtime errors and interruptions.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_SWAP.1.py'>Attack_SWAP.1.py</a></b></td>
      <td>- Facilitates an attack simulation using the SWAP method within the project's attack module<br>- It dynamically adjusts the system path to import necessary components and executes the attack with specified parameters, including device configuration and epoch settings<br>- This script is part of a broader effort to test and evaluate the robustness of systems against adversarial attacks, contributing to the project's security assessment objectives.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_FFT.py'>Attack_FFT.py</a></b></td>
      <td>- Facilitates an attack simulation using Fast Fourier Transform (FFT) within the project's framework<br>- It dynamically adjusts the system path to import necessary modules and configures attack parameters, including a correlation function<br>- The script executes the attack on a specified device, handling interruptions and runtime errors to ensure robustness<br>- This component is integral to testing the system's resilience against FFT-based attacks.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_COS_epoch1000.py'>Attack_COS_epoch1000.py</a></b></td>
      <td>- Facilitates adversarial attacks on machine learning models by leveraging the COS attack method<br>- It iterates over a set of predefined models and datasets, applying perturbations to evaluate model robustness<br>- The process dynamically adjusts batch sizes to manage GPU memory constraints and integrates results for comprehensive analysis<br>- This component is crucial for assessing and improving the security and resilience of the overall system.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_SWAP_L2.1.py'>Attack_SWAP_L2.1.py</a></b></td>
      <td>- Facilitates an attack simulation using the SWAPL2 method within the project's attack framework<br>- It dynamically adjusts the system path to import necessary modules and configures the attack with specific parameters, such as the number of epochs and a constant value<br>- The script is designed to execute the attack on a CUDA-enabled device, contributing to the project's broader goal of evaluating model robustness against adversarial attacks.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Test_100epoch.ipynb'>Test_100epoch.ipynb</a></b></td>
      <td>- Demonstrates the evaluation of attack strategies on a machine learning model without defense mechanisms<br>- Utilizes the Classifier_MACNN model to test the effectiveness of SWAP and COS attack methods on a univariate dataset over 1000 epochs<br>- Provides insights into attack success rates, failure counts, and distance metrics, contributing to the broader project goal of understanding model vulnerabilities and improving robustness.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_FFT.1.py'>Attack_FFT.1.py</a></b></td>
      <td>- Facilitates an attack simulation using Fast Fourier Transform (FFT) within the project's attack module<br>- It dynamically adjusts system paths to import necessary components and configures attack parameters, including correlation settings<br>- The script executes the attack process on a specified device, handling interruptions and runtime errors to ensure robustness<br>- This functionality is integral to testing the system's resilience against FFT-based attacks.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_COS.1.py'>Attack_COS.1.py</a></b></td>
      <td>- Facilitates an attack simulation using a cosine-based method within the broader project architecture<br>- It dynamically adjusts system paths to import necessary modules and executes the attack with specified parameters, leveraging GPU resources<br>- The code is part of a demonstration module focused on executing attacks without defensive measures, contributing to the project's exploration of attack strategies and their effectiveness.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_no_defence/Attack_SWAP_L2.py'>Attack_SWAP_L2.py</a></b></td>
      <td>- Facilitates an attack simulation using the SWAPL2 method within a machine learning context, specifically targeting scenarios without defense mechanisms<br>- It dynamically adjusts the system path to import necessary modules and executes the attack with specified parameters on a CUDA-enabled device<br>- This component is integral for testing the robustness of models against adversarial attacks in the broader project architecture.</td>
     </tr>
     </table>
    </blockquote>
   </details>
   <details>
    <summary><b>ATTACK_with_arg</b></summary>
    <blockquote>
     <table>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_with_arg/Attack_SWAP.py'>Attack_SWAP.py</a></b></td>
      <td>- Facilitates adversarial attacks on machine learning models by utilizing augmentation techniques like Gaussian noise and smoothing to enhance model robustness<br>- It orchestrates the training and attack processes, managing resources efficiently to prevent memory issues<br>- The code integrates with the broader architecture by leveraging existing utilities and models, ultimately contributing to the project's goal of improving model defense mechanisms against adversarial threats.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_with_arg/Attack_PGD.py'>Attack_PGD.py</a></b></td>
      <td>- Facilitates adversarial attacks on machine learning models using the Projected Gradient Descent (PGD) method<br>- It integrates augmentation techniques like Gaussian noise and smoothing to enhance model robustness<br>- The code iterates over various datasets and models, dynamically adjusting parameters to handle memory constraints, and ultimately evaluates the effectiveness of the attacks by aggregating performance metrics across different scenarios.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_with_arg/Attack_COS.py'>Attack_COS.py</a></b></td>
      <td>- The code in "Attack_COS.py" orchestrates adversarial attacks on machine learning models using augmentation techniques like Gaussian noise and smoothing<br>- It integrates with the project's training and defense mechanisms to evaluate model robustness across various datasets<br>- By leveraging CUDA for computation, it efficiently manages resources to handle potential memory constraints, ensuring comprehensive testing of model vulnerabilities against cosine-based attacks.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_with_arg/Attack_FFT.py'>Attack_FFT.py</a></b></td>
      <td>- The `Attack_FFT.py` module orchestrates the execution of adversarial attacks on machine learning models using Fourier Transform techniques<br>- It integrates augmentation and defense strategies to enhance model robustness against attacks<br>- By iterating over various datasets and models, it systematically applies noise and smoothing methods, optimizing attack parameters to evaluate and improve model resilience within the broader project architecture.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/ATTACK_with_arg/Attack_SWAP_L2.py'>Attack_SWAP_L2.py</a></b></td>
      <td>- Facilitates adversarial attacks on machine learning models by implementing augmentation techniques and defense strategies<br>- It orchestrates the training of models with noise and smoothing methods, then applies the SWAPL2 attack to evaluate model robustness<br>- The process involves handling CUDA memory constraints and managing attack parameters, ultimately aiming to enhance model resilience against adversarial perturbations across various datasets.</td>
     </tr>
     </table>
    </blockquote>
   </details>
   <details>
    <summary><b>Archived</b></summary>
    <blockquote>
     <table>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/test_train.ipynb'>test_train.ipynb</a></b></td>
      <td>- The notebook serves as a demonstration and testing environment for training models using the CorrelatedClassifierInception within the project's architecture<br>- It imports necessary modules, sets default training parameters, and provides examples of parameter handling and dictionary operations<br>- This facilitates experimentation and validation of model configurations, contributing to the overall development and refinement of the machine learning components in the codebase.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/train1.ipynb'>train1.ipynb</a></b></td>
      <td>- The file `Demo/Archived/train1.ipynb` serves as an introductory demonstration for the Trainer class within the project<br>- Its primary purpose is to guide users through the process of training and evaluating models, specifically focusing on the Inception-based architectures<br>- This notebook is part of the project's documentation and educational resources, aimed at helping users understand and effectively utilize the Trainer class in the broader context of the project's machine learning framework.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/deepfool.ipynb'>deepfool.ipynb</a></b></td>
      <td>- The file `deepfool.ipynb` located in the `Demo/Archived` directory serves as a demonstration of the DeepFool algorithm, which is used for generating adversarial examples in machine learning models<br>- This notebook is part of the project's archival demos, indicating that it provides historical or supplementary insights into the application of adversarial attacks within the broader context of the codebase<br>- Its primary purpose is to illustrate how the DeepFool method can be applied to evaluate and potentially improve the robustness of models against adversarial perturbations<br>- This aligns with the project's overarching goals of enhancing model security and reliability through comprehensive testing and analysis.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/train3.ipynb'>train3.ipynb</a></b></td>
      <td>- The file `train3.ipynb` located in the `Demo/Archived` directory serves as a historical reference for training models within the project<br>- It likely contains experimental or deprecated code that was used in earlier stages of model development<br>- This notebook provides insights into previous methodologies and configurations that were explored for training purposes<br>- Its presence in the `Archived` folder suggests that it is not part of the current active workflow but may still offer valuable context or lessons learned for ongoing or future model training efforts within the project's architecture.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/train2.ipynb'>train2.ipynb</a></b></td>
      <td>- The file `train2.ipynb` located in the `Demo/Archived` directory serves as an archived demonstration script within the broader project architecture<br>- Its primary purpose is to provide a historical reference or example of how training processes were previously executed in the project<br>- This notebook likely includes setup configurations and initializations necessary for running training tasks, as indicated by its inclusion of system path adjustments and imports<br>- It contributes to the project by offering insights into past methodologies and configurations, which can be useful for understanding the evolution of the project's training strategies or for replicating past experiments.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/train4.ipynb'>train4.ipynb</a></b></td>
      <td>- Demonstrates the functionality of the Trainer class by guiding users through training and evaluating an Inception-based classifier on time series datasets<br>- It facilitates model training, performance evaluation, and result consolidation<br>- Additionally, it manages logging, checkpointing, and metric consolidation, ensuring comprehensive tracking and storage of model weights and evaluation metrics across multiple datasets within the project architecture.</td>
     </tr>
     </table>
     <details>
      <summary><b>TEST6</b></summary>
      <blockquote>
       <table>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST6/train1.ipynb'>train1.ipynb</a></b></td>
        <td>- Facilitates the training and evaluation of a machine learning model using a specified dataset and device configuration<br>- It integrates a classifier model and adjusts learning rates dynamically during training<br>- The notebook also prepares for potential adversarial attacks by importing an attack model, although this functionality is currently commented out<br>- This contributes to the project's goal of developing robust and accurate predictive models.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST6/train1.2.ipynb'>train1.2.ipynb</a></b></td>
        <td>- Train1.2.ipynb orchestrates the training and evaluation of a machine learning model using a specified dataset<br>- It sets up the environment, selects the appropriate device for computation, and iterates over datasets to train a classifier model<br>- The notebook also includes provisions for adjusting learning rates and handling exceptions, contributing to the broader project goal of developing and refining correlation-based models.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST6/attack2.ipynb'>attack2.ipynb</a></b></td>
        <td>- Demonstrates the evaluation of a machine learning model's robustness against adversarial attacks within the project<br>- It utilizes a specific classifier to assess attack success rates and other metrics across various datasets<br>- The notebook integrates with the broader codebase by importing modules for training and attack simulation, contributing to the project's goal of enhancing model reliability and performance in adversarial scenarios.</td>
       </tr>
       </table>
      </blockquote>
     </details>
     <details>
      <summary><b>TEST3</b></summary>
      <blockquote>
       <table>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST3/train1.ipynb'>train1.ipynb</a></b></td>
        <td>- The file `train1.ipynb` located in `Demo/Archived/TEST3/` appears to serve as a preliminary or experimental notebook within the broader project architecture<br>- Its primary purpose is likely to facilitate testing or demonstration of specific functionalities related to the project's objectives, possibly involving data correlation or analysis, as suggested by the project name "CorrelationV0.1"<br>- The notebook's location in an "Archived" directory indicates that it might contain legacy code or previous iterations of experiments that have since been superseded or retained for reference<br>- This file contributes to the project by preserving historical context and insights that may inform future development or debugging efforts.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST3/train1.2.ipynb'>train1.2.ipynb</a></b></td>
        <td>- The file `train1.2.ipynb` located in `Demo/Archived/TEST3/` appears to be a Jupyter Notebook primarily used for setting up the environment for a project related to correlation analysis<br>- Its main purpose is to establish the working directory and ensure that the necessary paths are correctly configured for subsequent operations within the project<br>- This setup is crucial for maintaining consistency and accessibility of resources across different components of the codebase, particularly in a research or development setting where directory structures can be complex.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST3/attack2.ipynb'>attack2.ipynb</a></b></td>
        <td>- The file `attack2.ipynb` located in `Demo/Archived/TEST3/` serves as an exploratory or experimental notebook within the broader project architecture<br>- Its primary purpose is to facilitate testing or demonstration of specific functionalities or concepts, likely related to security or vulnerability assessments, given the context implied by the filename "attack2"<br>- This notebook is part of an archived section, suggesting it may contain legacy or experimental code that was used for testing purposes during earlier stages of development<br>- It imports core project modules, indicating its role in integrating and testing components from the main codebase.</td>
       </tr>
       </table>
      </blockquote>
     </details>
     <details>
      <summary><b>TEST4</b></summary>
      <blockquote>
       <table>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST4/train1.ipynb'>train1.ipynb</a></b></td>
        <td>- The file `train1.ipynb` located in `Demo/Archived/TEST4/` appears to serve as a preliminary or experimental script within the broader project architecture<br>- Its primary purpose is to set up the environment by determining and displaying the current working directory and home location paths<br>- This suggests that the notebook might be used for testing or demonstration purposes, particularly in configuring or verifying the setup before executing more complex tasks<br>- Given its placement in an "Archived" directory, it likely represents an older or deprecated version of a training or setup script, preserved for reference or historical context within the project.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST4/train1.2.ipynb'>train1.2.ipynb</a></b></td>
        <td>- The file `train1.2.ipynb` located in `Demo/Archived/TEST4/` appears to be part of a larger project focused on correlation analysis, as suggested by the project directory name `CorrelationV0.1`<br>- This Jupyter Notebook seems to serve as an exploratory or experimental script, likely used for testing or demonstrating specific functionalities related to the project's objectives<br>- It outputs directory paths, indicating its role in setting up or verifying the environment for executing further code<br>- Given its placement in an "Archived" folder, it might represent a previous iteration or a deprecated version of a training or testing process within the project's development lifecycle.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST4/attack2.ipynb'>attack2.ipynb</a></b></td>
        <td>- The file `attack2.ipynb` located in the `Demo/Archived/TEST4` directory appears to be part of a larger project that likely involves data analysis or machine learning, given its Jupyter Notebook format<br>- The primary purpose of this file is to demonstrate or test a specific aspect of the project, possibly related to security or robustness, as suggested by the name "attack." This could involve simulating or analyzing potential vulnerabilities or testing the resilience of a model or system<br>- The file is archived, indicating it may contain legacy code or experiments that are no longer actively developed but are retained for reference or historical purposes<br>- Overall, this notebook contributes to the project's broader goal of ensuring system integrity and performance under various conditions.</td>
       </tr>
       </table>
      </blockquote>
     </details>
     <details>
      <summary><b>Archived_KDD</b></summary>
      <blockquote>
       <details>
        <summary><b>attack</b></summary>
        <blockquote>
         <table>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/ATTACK_TS2Vec.py'>ATTACK_TS2Vec.py</a></b></td>
          <td>- Facilitates the execution of an attack simulation on a time-series classification model within the project's archived demo section<br>- By leveraging the TS2Vec framework, it orchestrates the training and evaluation of the Classifier_TS2V model over multiple iterations<br>- This process aids in assessing the model's robustness and performance, contributing to the broader objective of enhancing time-series analysis capabilities in the codebase.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/ATTACK_TS2Vec.1.py'>ATTACK_TS2Vec.1.py</a></b></td>
          <td>- Facilitates the execution of an attack simulation on the TS2Vec model within the project<br>- By importing necessary modules and setting up the environment, it leverages the `ATTACK_ALL` function to test the model's robustness over multiple runs<br>- This process aids in evaluating the model's performance under adversarial conditions, contributing to the overall goal of enhancing model reliability and security in the codebase.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/ATTACK_LstmFCN.py'>ATTACK_LstmFCN.py</a></b></td>
          <td>- Facilitates the execution of an attack simulation on a pre-trained LSTM-FCN model within the project's archived demo section<br>- By leveraging the ATTACK_ALL function, it aims to evaluate the model's robustness under specific conditions<br>- This component is integral to testing and validating the model's performance, contributing to the overall goal of enhancing the security and reliability of the machine learning models in the codebase.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/ce_kl_mix_attack.ipynb'>ce_kl_mix_attack.ipynb</a></b></td>
          <td>- Demonstrates an adversarial attack strategy using a mix of cross-entropy and KL divergence on a univariate dataset with the InceptionTime model<br>- It integrates training and attack modules to evaluate the model's robustness by perturbing data and measuring attack success rates and distances<br>- This approach aids in assessing and improving the model's resilience within the broader project framework.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/ATTACK_ResNet.py'>ATTACK_ResNet.py</a></b></td>
          <td>- Facilitates the execution of an attack simulation on a ResNet-18 model within the project's architecture<br>- By leveraging the ATTACK_ALL function, it integrates with the broader training and evaluation framework, utilizing GPU resources for enhanced performance<br>- This component is part of the archived demo section, indicating its role in testing or demonstrating attack scenarios on neural network models for research or educational purposes.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/deepfool_attack.py'>deepfool_attack.py</a></b></td>
          <td>- Facilitates adversarial attacks on machine learning models using the DeepFool algorithm within the project's architecture<br>- It iterates over specified models and datasets, configuring and executing the attack while managing computational resources<br>- This process aids in evaluating model robustness by generating adversarial examples, contributing to the overall goal of enhancing model security and reliability in the broader context of the project.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/ATTACK_MACNN.py'>ATTACK_MACNN.py</a></b></td>
          <td>- Facilitates the execution of an attack simulation on a MACNN-based classifier within the project's archived demo section<br>- By leveraging the ATTACK_ALL function, it tests the robustness of the MACNN model over multiple runs<br>- This contributes to the broader project architecture by ensuring the model's resilience and reliability, aligning with the project's focus on developing robust machine learning solutions.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/attack_macnn.ipynb'>attack_macnn.ipynb</a></b></td>
          <td>- Demonstrates the application of adversarial attack techniques on a machine learning model using the DeepFool method<br>- It integrates model training and attack execution within the project's framework, focusing on the "Beef" dataset<br>- The notebook evaluates the model's robustness by generating adversarial examples and measuring attack success rates, contributing to the broader goal of enhancing model resilience against adversarial threats in the codebase.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/ATTACK_Inception_Time.py'>ATTACK_Inception_Time.py</a></b></td>
          <td>- Facilitates the execution of an attack simulation on the Inception Time model within the project<br>- By leveraging the `ATTACK_ALL` function, it integrates with the broader architecture to test the robustness of the Inception Time classifier<br>- This process aids in evaluating the model's performance under adversarial conditions, contributing to the overall goal of enhancing model reliability and security in the system.</td>
         </tr>
         </table>
         <details>
          <summary><b>Archived</b></summary>
          <blockquote>
           <table>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_TS2Vec.2.py'>ATTACK_TS2Vec.2.py</a></b></td>
            <td>- Facilitates the execution of adversarial attacks on time series models within the project<br>- It orchestrates multiple attack methods across various datasets, leveraging GPU resources for efficient processing<br>- By integrating with the broader architecture, it supports model robustness evaluation by perturbing data and assessing the impact on model performance, contributing to the project's goal of enhancing time series model resilience.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_lstmfcn.1.py'>ATTACK_lstmfcn.1.py</a></b></td>
            <td>- Facilitates the evaluation of adversarial attack methods on univariate datasets using the LSTMFCN model<br>- It integrates attack strategies, iterates through datasets, and applies perturbations to assess model robustness<br>- The process includes logging and metric concatenation to provide insights into attack effectiveness, contributing to the broader project goal of enhancing model resilience against adversarial threats within the archived KDD context.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_Inception_Time.2.py'>ATTACK_Inception_Time.2.py</a></b></td>
            <td>- Facilitates the execution of adversarial attacks on time-series classification models using the Inception architecture<br>- It integrates various attack methods to evaluate model robustness across multiple datasets<br>- By leveraging a trainer for model preparation and a Mix class for attack execution, it systematically perturbs data and aggregates attack metrics, contributing to the project's focus on enhancing model security and performance analysis.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/attack_lstm_fcn.ipynb'>attack_lstm_fcn.ipynb</a></b></td>
            <td>- Demonstrates the implementation of an attack strategy using an LSTM-FCN model within the project's archived section<br>- It focuses on training and evaluating the model's performance on a univariate dataset, while also applying adversarial perturbations to assess the model's robustness<br>- This notebook serves as a historical reference for attack methodologies and their outcomes, contributing to the project's broader research on model vulnerabilities.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_TS2Vec.1.py'>ATTACK_TS2Vec.1.py</a></b></td>
            <td>- Facilitates the execution of adversarial attacks on time series models within the project, leveraging the TS2Vec framework<br>- It orchestrates multiple attack methods across various datasets, utilizing GPU resources efficiently<br>- The process involves perturbing datasets to evaluate model robustness, aggregating attack metrics for analysis<br>- This functionality is integral to assessing and enhancing the resilience of machine learning models against adversarial threats.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_ResNet.1.py'>ATTACK_ResNet.1.py</a></b></td>
            <td>- Facilitates adversarial attacks on a ResNet model by orchestrating multiple attack methods across various datasets<br>- Utilizes a distributed GPU setup to efficiently manage computational resources<br>- Integrates with the broader architecture by leveraging shared components for model training and attack execution, ultimately contributing to the evaluation and robustness testing of machine learning models within the project.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_ResNet.2.py'>ATTACK_ResNet.2.py</a></b></td>
            <td>- Facilitates adversarial attacks on a ResNet model by orchestrating multiple attack methods across various datasets<br>- Utilizes GPU resources efficiently by distributing tasks based on GPU availability<br>- Integrates with the broader codebase to enhance model robustness evaluation through systematic perturbation and metric aggregation<br>- Supports iterative experimentation by allowing multiple runtime configurations and parameter adjustments, contributing to the project's focus on model resilience and security.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_Inception_Time.1.py'>ATTACK_Inception_Time.1.py</a></b></td>
            <td>- Facilitates the execution of adversarial attacks on time-series classification models using the Inception architecture<br>- It leverages predefined attack methods to perturb datasets and evaluates the robustness of the model<br>- By iterating over multiple datasets and attack configurations, it aims to assess and enhance the model's resilience against adversarial inputs, contributing to the project's focus on improving model security and reliability.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_macnn.1.py'>ATTACK_macnn.1.py</a></b></td>
            <td>- Facilitates the evaluation of adversarial attack methods on univariate datasets within the project<br>- It leverages a pre-trained MACNN model to generate and assess adversarial perturbations using various attack strategies<br>- The process iterates over multiple datasets and attack configurations, aiming to measure the robustness and effectiveness of the model against adversarial inputs, contributing to the project's overarching goal of enhancing model security and reliability.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/attack_Inception_Time.ipynb'>attack_Inception_Time.ipynb</a></b></td>
            <td>- Demonstrates an attack simulation on time-series data using the InceptionTime model within the project's framework<br>- It integrates the Mix attack method to perturb datasets and evaluates the model's robustness against adversarial attacks<br>- The notebook serves as an archived demonstration of attack strategies, contributing to the project's goal of enhancing model resilience and security in time-series classification tasks.</td>
           </tr>
           <tr>
            <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/attack/Archived/ATTACK_lstmfcn.2.py'>ATTACK_lstmfcn.2.py</a></b></td>
            <td>- Facilitates the evaluation of adversarial attack methods on univariate datasets using an LSTM-FCN model<br>- It integrates attack strategies from a predefined set, applies them to datasets, and assesses their impact on model performance<br>- The process involves iterating through datasets, executing attacks, and compiling metrics to understand the robustness of the model against various adversarial perturbations within the project's architecture.</td>
           </tr>
           </table>
          </blockquote>
         </details>
        </blockquote>
       </details>
       <details>
        <summary><b>train</b></summary>
        <blockquote>
         <table>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_TS2Vec.1.ipynb'>train_TS2Vec.1.ipynb</a></b></td>
          <td>- The Jupyter notebook in the archived section of the project is designed to train and evaluate a time series classification model using the TS2Vec framework<br>- It integrates with the broader codebase by importing necessary modules and setting up a training pipeline<br>- The notebook demonstrates model performance metrics such as accuracy and F1 score, contributing to the project's goal of developing robust time series analysis tools.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_TS2Vec.1.py'>train_TS2Vec.1.py</a></b></td>
          <td>- Facilitates the training and evaluation of a time series classification model using the TS2Vec framework within the project's architecture<br>- It iterates over a set of univariate datasets, configuring and executing a training process with specific parameters and optimizations<br>- The results are aggregated for both training and testing phases, contributing to the overall performance assessment and model refinement in the broader project context.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_lstm_fcn.py'>train_lstm_fcn.py</a></b></td>
          <td>- Facilitates the training and evaluation of a Long Short-Term Memory Fully Convolutional Network (LSTM-FCN) model on univariate datasets<br>- Integrates with the broader codebase by leveraging shared modules and datasets, and utilizes GPU acceleration for enhanced performance<br>- Contributes to the project's goal of developing robust time-series analysis models by managing training processes and aggregating performance metrics for both training and testing phases.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_ResNet.1.ipynb'>train_ResNet.1.ipynb</a></b></td>
          <td>- The notebook facilitates the training and evaluation of a ResNet-18 model on various univariate datasets<br>- It integrates with the broader project by leveraging a custom Trainer class to manage the training process, including learning rate adjustments and performance metric aggregation<br>- This contributes to the project's goal of developing robust machine learning models for time-series data analysis.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_macnn1.py'>train_macnn1.py</a></b></td>
          <td>- Facilitates the training and evaluation of a machine learning model using the MACNN architecture on a set of univariate datasets<br>- It orchestrates the training process, including setting hyperparameters and optimizing the model, and aggregates performance metrics for both training and testing phases<br>- This contributes to the broader project by enabling model experimentation and performance tracking within the archived KDD dataset context.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_transformer.ipynb'>train_transformer.ipynb</a></b></td>
          <td>- The file `train_transformer.ipynb` located in `Demo/Archived/Archived_KDD/train/` serves as a historical artifact within the project, likely capturing an earlier approach or experiment related to training a transformer model for the KDD2024 initiative<br>- Its placement in the "Archived" directory suggests that it is not part of the current active development but may provide valuable insights or reference for understanding past methodologies or for future retrospectives<br>- The notebook's primary purpose is to document and execute the training process of a transformer model, contributing to the project's broader goal of advancing machine learning capabilities for the KDD2024 objectives.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_lstm_fcn.1.ipynb'>train_lstm_fcn.1.ipynb</a></b></td>
          <td>- The file `train_lstm_fcn.1.ipynb` located in `Demo/Archived/Archived_KDD/train/` serves as a historical artifact within the project, showcasing an early implementation of a training routine for a model combining Long Short-Term Memory (LSTM) networks and Fully Convolutional Networks (FCN)<br>- Its primary purpose is to document and preserve the initial experimental setup and methodologies used in the project's development phase<br>- This notebook is part of the archived section, indicating that it may not be actively maintained or used in the current iteration of the project but provides valuable insights into the evolution of the model training strategies.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_ResNet.1.py'>train_ResNet.1.py</a></b></td>
          <td>- Facilitates the training and evaluation of a ResNet18 model on various univariate datasets within the project<br>- It leverages a Trainer class to manage the training process, including setting hyperparameters and optimizing the model<br>- The results are aggregated for both training and testing phases, contributing to the project's goal of developing robust machine learning models for univariate data analysis.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_Inception_Time.2.ipynb'>train_Inception_Time.2.ipynb</a></b></td>
          <td>- The file `train_Inception_Time.2.ipynb` located in `Demo/Archived/Archived_KDD/train/` serves as a historical artifact within the project, likely used for training a machine learning model using the Inception Time architecture<br>- Its placement in the "Archived" directory suggests that it is not part of the current active development but may have been instrumental in earlier phases of the project<br>- This notebook is part of a larger effort to explore or validate time-series classification techniques, contributing to the project's overarching goal of advancing predictive modeling capabilities.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_Inception_Time.1.ipynb'>train_Inception_Time.1.ipynb</a></b></td>
          <td>- The file `train_Inception_Time.1.ipynb` located in `Demo/Archived/Archived_KDD/train/` is part of an archived section of the project, indicating it may contain legacy or experimental code<br>- Its primary purpose is to facilitate the training of models using the Inception Time architecture, which is likely a deep learning model tailored for time series data<br>- This notebook is part of the broader project aimed at developing and refining machine learning models for the KDD2024 competition or initiative<br>- By being in the archived directory, it suggests that this particular implementation might have been an earlier version or a specific experiment that contributed to the project's evolution<br>- The notebook's role is to serve as a historical reference or a potential resource for revisiting past methodologies within the project's lifecycle.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_macnn.1.ipynb'>train_macnn.1.ipynb</a></b></td>
          <td>- The Jupyter notebook in the Demo/Archived/Archived_KDD directory serves as a training script for machine learning models using PyTorch<br>- It focuses on evaluating the performance of the MACNN and LSTMFCN models on univariate datasets<br>- The script sets up the training environment, configures model parameters, and outputs performance metrics such as accuracy, precision, and recall, contributing to the project's model evaluation and benchmarking efforts.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_Inception_Time.py'>train_Inception_Time.py</a></b></td>
          <td>- Facilitates the training and evaluation of the Inception Time model on univariate datasets within the project<br>- It initializes a trainer for each dataset, specifying parameters like epochs and device usage, and manages the method path for results<br>- Additionally, it consolidates training and testing metrics, contributing to the project's goal of implementing and assessing time-series classification models.</td>
         </tr>
         <tr>
          <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/Archived_KDD/train/train_lstm_fcn.2.ipynb'>train_lstm_fcn.2.ipynb</a></b></td>
          <td>- The file "train_lstm_fcn.2.ipynb" located in the "Demo/Archived/Archived_KDD/train" directory is part of a larger project architecture focused on data analysis or machine learning, likely related to the KDD (Knowledge Discovery and Data Mining) domain<br>- This specific notebook appears to be an archived script for training a model using a combination of Long Short-Term Memory (LSTM) networks and Fully Convolutional Networks (FCN)<br>- Its primary purpose is to facilitate the training process of this hybrid model, which is likely used for time-series analysis or sequence prediction tasks within the project<br>- The notebook is structured to set up the environment, import necessary modules, and configure logging, indicating its role in preparing and executing model training experiments.</td>
         </tr>
         </table>
        </blockquote>
       </details>
      </blockquote>
     </details>
     <details>
      <summary><b>TEST2</b></summary>
      <blockquote>
       <table>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST2/train1.ipynb'>train1.ipynb</a></b></td>
        <td>- Facilitates the training and evaluation of a machine learning model using a specified dataset and device configuration<br>- Integrates with the broader project by importing necessary modules and setting up the environment for model training<br>- Additionally, includes functionality for adjusting learning rates and handling potential interruptions, contributing to the project's goal of developing and testing correlation-based models efficiently.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST2/train1.2.ipynb'>train1.2.ipynb</a></b></td>
        <td>- Demonstrates a training process for machine learning models using the PyTorch framework, focusing on univariate datasets<br>- It integrates a model classifier and a training loop to evaluate model performance, adjusting learning rates dynamically<br>- The notebook also prepares for potential adversarial attacks by importing relevant modules, indicating its role in both model training and robustness testing within the project's architecture.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST2/attack2.ipynb'>attack2.ipynb</a></b></td>
        <td>- The file `attack2.ipynb` located in `Demo/Archived/TEST2/` serves as an exploratory or experimental notebook within the broader project architecture<br>- Its primary purpose is to test or demonstrate specific functionalities or concepts related to the project's objectives, likely involving correlation analysis given the project name<br>- As it resides in an "Archived" directory, it may contain legacy code or preliminary experiments that contributed to the development of the main features in the project<br>- This notebook is not central to the core functionality but provides valuable insights or historical context for the project's evolution.</td>
       </tr>
       </table>
      </blockquote>
     </details>
     <details>
      <summary><b>TEST5</b></summary>
      <blockquote>
       <table>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST5/train1.ipynb'>train1.ipynb</a></b></td>
        <td>- The file `train1.ipynb` located in `Demo/Archived/TEST5/` appears to be a Jupyter Notebook that is part of a larger project focused on correlation analysis, as suggested by the project path<br>- The primary purpose of this notebook is likely to serve as an experimental or testing ground for code related to the project's objectives<br>- It may include initial data exploration, testing of algorithms, or validation of concepts before they are integrated into the main codebase<br>- Given its location in an "Archived" directory, it might contain legacy code or previous iterations of experiments that have been superseded by more recent developments in the project.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST5/train1.2.ipynb'>train1.2.ipynb</a></b></td>
        <td>- The file `train1.2.ipynb` located in `Demo/Archived/TEST5/` appears to be a Jupyter Notebook used for exploratory data analysis or experimentation within the broader project<br>- Its primary purpose is likely to facilitate testing and development of features related to the project's core functionality, which involves correlation analysis<br>- By being situated in an "Archived" directory, it suggests that this notebook might contain legacy code or preliminary experiments that contributed to the project's development but are not part of the active codebase<br>- This file serves as a historical reference or a resource for understanding past approaches and methodologies used in the project.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST5/attack2.ipynb'>attack2.ipynb</a></b></td>
        <td>- The file `attack2.ipynb` located in `Demo/Archived/TEST5/` serves as an exploratory or experimental notebook within the broader project architecture<br>- Its primary purpose is to facilitate testing and demonstration of specific functionalities related to the project's objectives, likely involving correlation analysis given the project context<br>- The notebook appears to be part of an archived section, suggesting it may contain legacy or experimental code that was used to test certain hypotheses or features during the development phase<br>- It contributes to the project by providing insights or validation for certain approaches, although it may not be part of the active codebase used in production.</td>
       </tr>
       </table>
      </blockquote>
     </details>
     <details>
      <summary><b>TEST</b></summary>
      <blockquote>
       <table>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST/train1.ipynb'>train1.ipynb</a></b></td>
        <td>- The file `train1.ipynb` located in `Demo/Archived/TEST` appears to be a Jupyter Notebook used for testing or demonstration purposes within the broader project<br>- Its primary role is to set up the environment by configuring system paths, which suggests it might be used to ensure that the necessary directories and dependencies are correctly referenced for running experiments or demonstrations<br>- This setup is crucial for maintaining consistency and reproducibility in the project's workflow, especially when dealing with complex directory structures or multiple modules<br>- The notebook's placement in an "Archived" folder indicates it might be part of legacy code or previous iterations of the project, serving as a reference or backup for current development efforts.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST/attack1.ipynb'>attack1.ipynb</a></b></td>
        <td>- The file `attack1.ipynb` located in the `Demo/Archived/TEST` directory appears to be an experimental or exploratory Jupyter Notebook intended for testing purposes within the project<br>- Its primary purpose is likely to serve as a sandbox for developing and validating attack scenarios or testing specific functionalities in isolation<br>- Given its placement in an "Archived" and "TEST" folder, it suggests that the notebook is not part of the active codebase but rather a historical or deprecated artifact that might have been used during earlier stages of development or for educational purposes<br>- This file does not contribute directly to the core functionality of the project but may provide insights or reference for future testing or development efforts.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST/train2.ipynb'>train2.ipynb</a></b></td>
        <td>- The file `train2.ipynb` located in `Demo/Archived/TEST/` serves as an experimental or archived notebook within the broader project architecture<br>- Its primary purpose is to document and test various training processes or methodologies related to the project's objectives, likely focusing on correlation analysis given the project name<br>- As it resides in an "Archived" directory, it suggests that the notebook contains legacy or exploratory work that may have informed current models or approaches but is not part of the active codebase<br>- This file is useful for historical reference or for understanding the evolution of the project's training strategies.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST/test.ipynb'>test.ipynb</a></b></td>
        <td>- Demonstrates testing procedures for archived components within the project, ensuring that legacy functionalities remain intact and reliable<br>- By focusing on test cases and validation, it supports the overall codebase architecture by maintaining backward compatibility and providing a reference for future updates<br>- This contributes to the project's robustness by safeguarding against regressions in older modules.</td>
       </tr>
       <tr>
        <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/Archived/TEST/attack2.ipynb'>attack2.ipynb</a></b></td>
        <td>- The file `attack2.ipynb` located in `Demo/Archived/TEST/` serves as an exploratory or experimental notebook within the broader project architecture<br>- Its primary purpose is to facilitate testing and demonstration of specific functionalities or scenarios, likely related to security or vulnerability assessments, given the context implied by the filename "attack2"<br>- This notebook is part of the archived section, suggesting it may contain legacy or experimental code that was used for testing purposes during earlier stages of development<br>- It contributes to the project by providing a space for trial implementations and insights that can inform more stable and integrated solutions within the main codebase.</td>
       </tr>
       </table>
      </blockquote>
     </details>
    </blockquote>
   </details>
   <details>
    <summary><b>TEST</b></summary>
    <blockquote>
     <table>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/TEST/test_normal.ipynb'>test_normal.ipynb</a></b></td>
      <td>- The file `test_normal.ipynb` located in the `Demo/TEST` directory serves as a setup and configuration script within the broader architecture of the project<br>- Its primary purpose is to dynamically adjust the Python path to ensure that the necessary modules and packages are accessible for testing and demonstration purposes<br>- By iteratively appending directories to the system path, it facilitates the import of project-specific initializations, thereby supporting the seamless execution of test cases and demos<br>- This functionality is crucial for maintaining modularity and flexibility across different environments within the project.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/TEST/test_cosine.ipynb'>test_cosine.ipynb</a></b></td>
      <td>- Demonstrates the testing of adversarial attack models within the project, focusing on evaluating different attack strategies such as SWAP, SWAPL2, and COS<br>- It assesses the models' performance on a dataset, providing metrics like attack success rate and distance measures<br>- This testing process is crucial for validating the robustness and effectiveness of the adversarial training methods implemented in the broader codebase.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/TEST/get_all_train.ipynb'>get_all_train.ipynb</a></b></td>
      <td>- Facilitates the training and evaluation of various machine learning models for time series classification within the project<br>- It integrates data augmentation techniques and defense mechanisms to enhance model robustness<br>- The notebook orchestrates the training process, manages output directories, and consolidates training and testing metrics, contributing to the project's goal of developing effective time series classification models with improved performance and reliability.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/TEST/test_Argunment.ipynb'>test_Argunment.ipynb</a></b></td>
      <td>- The notebook facilitates testing and demonstration of data augmentation techniques within the project's architecture<br>- It dynamically adjusts the Python path to import necessary modules and utilizes various augmentation methods from the Augmentation class<br>- Additionally, it sets up and trains a defense model using the Inception Time architecture, applying Gaussian noise augmentation to evaluate model robustness across different standard deviations.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/TEST/test_correlation.ipynb'>test_correlation.ipynb</a></b></td>
      <td>- The file `Demo/TEST/test_correlation.ipynb` serves as a testing script within the project's architecture, specifically designed to validate and demonstrate the correlation functionalities of the system<br>- It is part of the broader testing suite located in the `Demo/TEST` directory, which is likely dedicated to ensuring the reliability and accuracy of various components within the project<br>- This notebook is instrumental in providing a hands-on, interactive environment to verify that the correlation features perform as expected, thereby supporting the overall integrity and robustness of the codebase.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/TEST/test_utils.ipynb'>test_utils.ipynb</a></b></td>
      <td>- The file `Demo/TEST/test_utils.ipynb` serves as a testing utility within the project's architecture<br>- Its primary purpose is to facilitate the validation and verification of various components or functionalities in the codebase<br>- By providing a structured environment for executing test cases, it ensures that the project's features perform as expected and helps maintain code reliability and quality<br>- This file is integral to the project's testing framework, supporting continuous integration and development practices.</td>
     </tr>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Demo/TEST/test_fft.ipynb'>test_fft.ipynb</a></b></td>
      <td>- The file `test_fft.ipynb` located in the `Demo/TEST` directory serves as a demonstration and testing ground for the Fast Fourier Transform (FFT) functionality within the project<br>- Its primary purpose is to validate and showcase how FFT is implemented and utilized in the broader codebase<br>- This notebook likely includes examples, test cases, and visualizations that help users understand the FFT process and verify its correctness and performance<br>- By doing so, it ensures that the FFT component integrates seamlessly with the rest of the project's architecture, maintaining the overall system's reliability and efficiency.</td>
     </tr>
     </table>
    </blockquote>
   </details>
  </blockquote>
 </details>
 <details> <!-- Train Submodule -->
  <summary><b>Train</b></summary>
  <blockquote>
   <table>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Train/trainer.py'>trainer.py</a></b></td>
    <td>- The `Trainer` class orchestrates the training and evaluation processes within the codebase, managing model initialization, data loading, and training configurations<br>- It supports adversarial training and model checkpointing, ensuring robust training workflows<br>- By handling loss calculations, optimizer updates, and performance metrics, it facilitates efficient model training and evaluation, contributing to the overall machine learning pipeline of the project.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Train/lstm_fcn.py'>lstm_fcn.py</a></b></td>
    <td>- The `lstm_fcn.py` module implements a neural network model combining Long Short-Term Memory (LSTM) and Fully Convolutional Network (FCN) architectures<br>- It is designed to process sequential data, leveraging LSTM for temporal dependencies and convolutional layers for feature extraction<br>- This hybrid model aims to enhance classification tasks within the project by integrating both temporal and spatial data representations.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Train/macnn.py'>macnn.py</a></b></td>
    <td>- Implements a multi-attention convolutional neural network (MACNN) classifier designed to process input data and predict class labels<br>- It utilizes multiple convolutional layers with varying kernel sizes and an attention mechanism to enhance feature extraction<br>- This component is integral to the project's architecture, focusing on improving classification accuracy by leveraging advanced neural network techniques for complex data analysis.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Train/transformer.py'>transformer.py</a></b></td>
    <td>- The `Train/transformer.py` file implements a Transformer-based model designed for classification tasks within the project<br>- It incorporates positional encoding and a Transformer encoder to process sequential data, applying dropout and ReLU activation for regularization and non-linearity<br>- The model concludes with a fully connected layer to output class predictions, contributing to the project's machine learning capabilities by enabling sequence classification.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Train/inception_time.py'>inception_time.py</a></b></td>
    <td>- The `Train/inception_time.py` file implements the Inception architecture for time series classification within the project<br>- It defines the InceptionModule and Classifier_INCEPTION classes, which are responsible for constructing and managing the deep learning model's layers, including bottleneck and shortcut connections<br>- The file also incorporates augmentation strategies and defensive mechanisms to enhance model robustness and adaptability.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Train/defence.py'>defence.py</a></b></td>
    <td>- The Defence class integrates data augmentation into the model training process, enhancing the robustness of the mother model by applying specified augmentation techniques before passing data through the model<br>- This approach supports the overall architecture by enabling flexible augmentation strategies, which can be tailored through parameters, thereby improving the model's ability to generalize and perform effectively in diverse scenarios.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Train/resnet.py'>resnet.py</a></b></td>
    <td>- Implements a ResNet-based architecture for training neural networks, focusing on building blocks like ResRoad, MainRoad, and ResNetBlock to facilitate residual learning<br>- The ClassifierResNet and its subclass, ClassifierResNet18, are designed for classification tasks, supporting both shallow and deep configurations<br>- This module is integral to the project's machine learning component, enabling efficient feature extraction and classification.</td>
   </tr>
   </table>
   <details>
    <summary><b>TS_2_V</b></summary>
    <blockquote>
     <table>
     <tr>
      <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Train/TS_2_V/main.py'>main.py</a></b></td>
      <td>- The file `Train/TS_2_V/main.py` is a crucial component of the project's architecture, primarily focused on defining convolutional neural network (CNN) layers tailored for time-series data processing<br>- It introduces custom convolutional blocks, such as `SamePadConv` and `ConvBlock`, which are designed to handle input sequences with specific padding and dilation configurations<br>- These components are likely used to enhance the model's ability to capture temporal patterns and dependencies in the data, contributing to the overall goal of the project, which appears to involve advanced time-series analysis or prediction.</td>
     </tr>
     </table>
   </details>
  </blockquote>
 </details>
 <details> <!-- Config Submodule -->
  <summary><b>Config</b></summary>
  <blockquote>
   <table>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Config/full_pramater_train.jsonc'>full_pramater_train.jsonc</a></b></td>
    <td>- The configuration file serves as a detailed guide for setting up training parameters and data augmentation techniques within the project<br>- It outlines the dataset, device selection, batch size, epochs, and loss function, while also providing options for data augmentation and model defense strategies<br>- This file aids users in customizing and analyzing different parameter choices to optimize model training and performance.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Config/DEFAULT_ATTACK.json'>DEFAULT_ATTACK.json</a></b></td>
    <td>- Config/DEFAULT_ATTACK.json defines the default configuration parameters for executing adversarial attacks within the project<br>- It specifies settings such as the dataset, model type, batch size, and various attack parameters like epsilon values and loss functions<br>- This configuration file plays a crucial role in standardizing attack setups, ensuring consistency and reproducibility across different experiments and facilitating easier adjustments for testing and development.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Config/DEFAULT_TRAIN.json'>DEFAULT_TRAIN.json</a></b></td>
    <td>- The configuration file defines parameters for training a machine learning model within the project<br>- It specifies the dataset, device, batch size, number of epochs, and loss function, among other settings<br>- It also includes options for data normalization, model selection, and defensive strategies against adversarial attacks<br>- This setup facilitates consistent and customizable training processes, aligning with the project's broader goals of model development and experimentation.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Config/DEFAULT_DATA_NAME.json'>DEFAULT_DATA_NAME.json</a></b></td>
    <td>- The JSON configuration file serves as a centralized repository of dataset names used within the project<br>- It provides a comprehensive list of datasets that can be referenced throughout the codebase, facilitating data management and ensuring consistency<br>- This setup supports modularity and scalability by allowing easy updates and integration of new datasets without altering the core architecture of the project.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Config/full_pramater_attack.jsonc'>full_pramater_attack.jsonc</a></b></td>
    <td>- The configuration file outlines parameters for conducting adversarial attacks on machine learning models, specifically targeting the Classifier_INCEPTION model<br>- It provides a detailed explanation of various attack settings, including perturbation limits, target selection methods, and loss functions<br>- The file serves as a guide for users to customize and analyze different attack strategies within the project's adversarial attack framework.</td>
   </tr>
   </table>
  </blockquote>
 </details>
 <details> <!-- Attack Submodule -->
  <summary><b>Attack</b></summary>
  <blockquote>
   <table>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/correlation.py'>correlation.py</a></b></td>
    <td>- Facilitates the calculation of weighted correlation coefficients for data analysis within the attack module of the project<br>- It provides classes and methods to compute correlations using different weighting functions, such as Gaussian and Sigmoid, and integrates these calculations into adversarial attack strategies<br>- This enhances the robustness and effectiveness of the attack methods by incorporating correlation-based regularization into the loss functions.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/mix.py'>mix.py</a></b></td>
    <td>- Facilitates adversarial attacks on machine learning models by implementing a mix of attack strategies, including gradient-based and Kullback-Leibler divergence methods<br>- It configures attack parameters, manages target selection, and calculates loss functions to generate adversarial examples<br>- This functionality is crucial for testing model robustness within the project's architecture, ensuring models can withstand adversarial perturbations and improve their defensive capabilities.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/swap.py'>swap.py</a></b></td>
    <td>- Implements the SWAP attack strategy within the project's attack module, leveraging the Mix class to facilitate adversarial attacks with specific configurations<br>- Establishes a directory structure for output storage based on attack parameters and dataset, integrating seamlessly with the broader architecture to enhance the project's capabilities in generating and managing adversarial examples for testing and evaluation purposes.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/cosine.py'>cosine.py</a></b></td>
    <td>- Facilitates the computation of cosine similarity between tensors and integrates this metric into an attack strategy within the project's architecture<br>- By leveraging cosine similarity, it enhances the effectiveness of adversarial attacks, particularly in optimizing loss functions<br>- This approach contributes to the broader goal of improving model robustness and evaluating vulnerabilities in machine learning models through sophisticated attack methodologies.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/deepfool.py'>deepfool.py</a></b></td>
    <td>- DeepFool class implements an adversarial attack algorithm designed to generate perturbed inputs that deceive machine learning models<br>- By iteratively adjusting input data, it aims to find minimal perturbations that cause model misclassification<br>- This functionality is crucial for evaluating model robustness within the project's architecture, providing insights into model vulnerabilities and helping improve defenses against adversarial attacks.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/fft.py'>fft.py</a></b></td>
    <td>- The `Attack/fft.py` module enhances the project's attack capabilities by implementing weighted Fast Fourier Transform (FFT) techniques<br>- It introduces classes for applying Gaussian and Sigmoid weighted FFTs to input data, facilitating advanced attack strategies<br>- By integrating these FFT methods with loss functions, the module contributes to generating adversarial examples, optimizing attack effectiveness, and improving the robustness evaluation of machine learning models within the codebase.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/swap_l2.py'>swap_l2.py</a></b></td>
    <td>- Facilitates adversarial attack generation within the project by implementing the SWAPL2 attack method<br>- It extends the Mix class to create adversarial examples using L2 regularization and a specified loss function<br>- The code integrates with the project's architecture to output results in a structured directory, supporting the evaluation and analysis of model robustness against adversarial perturbations.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/pgd.py'>pgd.py</a></b></td>
    <td>- Implements the Projected Gradient Descent (PGD) attack method within the broader architecture, focusing on adversarial attack strategies<br>- It extends the Mix class to configure specific attack parameters and manages output paths for results<br>- This integration supports the project's goal of enhancing model robustness by simulating adversarial scenarios, contributing to the overall security and reliability of the machine learning models.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Attack/attacker.py'>attacker.py</a></b></td>
    <td>- The `Attack` class orchestrates adversarial attacks on machine learning models by loading model weights, preparing datasets, and executing perturbations to evaluate model robustness<br>- It calculates metrics such as attack success rate and distance measures, and provides functionalities for saving results and visualizing comparisons between original and perturbed samples<br>- This component is integral for assessing and enhancing model defenses within the project.</td>
   </tr>
   </table>
  </blockquote>
 </details>
 <details> <!-- Utils Submodule -->
  <summary><b>Utils</b></summary>
  <blockquote>
   <table>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Utils/constant.py'>constant.py</a></b></td>
    <td>- The `Utils/constant.py` file establishes foundational configurations and logging mechanisms for the project<br>- It defines paths for datasets, models, and outputs, sets up logging for tracking execution, and specifies default parameters for training and attack processes<br>- Additionally, it manages device allocation for computations and maintains a dictionary of model classifiers, ensuring a structured and consistent environment for the project's operations.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Utils/augmentation.py'>augmentation.py</a></b></td>
    <td>- Facilitates data augmentation for machine learning models by providing a suite of methods to apply various transformations to input data<br>- These transformations include jittering, masking, adding noise, smoothing, and calculating correlations, which enhance model robustness and performance by simulating diverse data scenarios<br>- The class also offers utility functions to retrieve available augmentation methods, supporting dynamic integration into the broader project architecture.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Utils/correlation.py'>correlation.py</a></b></td>
    <td>- Facilitates efficient feature extraction by implementing a parallel dilated convolutional layer, which enhances the model's ability to capture multi-scale temporal patterns in input data<br>- This component is integral to the project's architecture, supporting advanced neural network operations by combining outputs from multiple dilated convolutions, thereby improving the overall performance and accuracy of the system in processing sequential data.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Utils/package.py'>package.py</a></b></td>
    <td>- Facilitates utility functions and imports essential libraries for data manipulation, machine learning, and deep learning tasks within the project<br>- Enhances functionality by integrating data processing, model evaluation, and optimization capabilities<br>- Supports seamless interaction with various data formats and structures, while ensuring compatibility with machine learning frameworks like PyTorch<br>- Contributes to the project's modularity and scalability by centralizing common dependencies and utilities.</td>
   </tr>
   <tr>
    <td><b><a href='https://github.com/David-Lzy/ADMA2024/blob/master/Utils/utils.py'>utils.py</a></b></td>
    <td>- The `Utils/utils.py` file serves a crucial role in the project's architecture by providing utility functions for data loading and preprocessing<br>- Its primary purpose is to facilitate the efficient handling of datasets by reading, normalizing, and mapping labels for different phases of data processing, such as training<br>- This utility ensures that data is readily accessible and correctly formatted, supporting the broader data pipeline within the project<br>- By abstracting these operations, it enhances code reusability and maintainability across the codebase.</td>
   </tr>
   </table>
  </blockquote>
 </details>
</details>

---

## Getting Started

### Prerequisites

Before getting started with ADMA2024, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python > 3.9
- **pytorch** > 2.0

  Note: you can easily make the anaconda env after clone the project.

### Installation

Install ADMA2024 using one of the following methods:

**Build from source:**

1. manually mkdir file in you workSpace:

```sh
❯ mkdir DATA
❯ mkdir LOG
❯ mkdir OUTPUT
```

1. Clone the ADMA2024 repository and rename:

```sh
❯ git clone https://github.com/David-Lzy/ADMA2024/
❯ mv ADMA2024 CODE
```

2. Navigate to the CODE directory:

```sh
❯ cd CODE
```

3. Install the project dependencies:

```sh
❯ cd ENV
❯ conda env create -f freeze.yml
```

---

## Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## Contributing

- **💬 [Join the Discussions](https://github.com/David-Lzy/ADMA2024/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/David-Lzy/ADMA2024/issues)**: Submit bugs found or log feature requests for the `ADMA2024` project.
- **💡 [Submit Pull Requests](https://github.com/David-Lzy/ADMA2024/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.

   ```sh
   git clone https://github.com/David-Lzy/ADMA2024/
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
   <a href="https://github.com{/David-Lzy/ADMA2024/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=David-Lzy/ADMA2024">
   </a>
</p>
</details>

---

## License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- Additional tanks to Project [readme-ai](https://github.com/eli64s/readme-ai) to help make this readme file. Note: **only this readme file helped by "readme-ai", not including the other codes.**

---
