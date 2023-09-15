# Alzheimer Classification based on <br> Multi-Task Event-Specific EEG-fNIRS Feature Fusion
<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
## Table of Contents
<ol>
  <li><a href="#abstract">Abstract</a></li>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#prerequisites">Prerequisites</a></li>
      <li><a href="#how-to-run">How to run</a></li>
    </ul>
  </li>
  <li><a href="#experimental-results">Experimental Results</a></li>
  <li><a href="#contact">Contact</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>



<!-- ABSTRACT -->
## Abstract
As the population ages, the number of people with Alzheimer's disease is increasing dramatically. However, the use of functional Magnetic Resonance Imaging (fMRI) as a method for Alzheimer's diagnosis has several challenges. Its high cost can limit accessibility, the process is time-consuming, and physical discomfort experienced during the procedure often leads to reluctance among potential patients. Hence, recent studies have shifted towards more cost-effective, time-efficient, portable, and motion-insensitive tools such as Electroencephalography (EEG) and functional Near-Infrared Spectroscopy (fNIRS) for diagnosing Alzheimer's disease.
The aim of this study is to use both EEG and fNIRS signal data collected through four simple tasks (resting state, oddball task, 1-back task, verbal fluency task) for Alzheimer classification, and to present an event-specific feature extraction method and feature selection method suitable for the data. EEG and fNIRS signals were collected from 144 subjects including 63 Healthy Controls (HC), 46 patients with Mild Cognitive Impairment (MCI), and 35 patients with Alzheimer's Disease (AD).
Through our proposed event-specific feature extraction method, we extracted distinct features from each EEG and fNIRS signal, and the Recursive Feature Elimination with Cross-Validation (RFECV) algorithm was utilized to select hybrid EEG-fNIRS features useful for Alzheimer classification. The finally selected features achieved high performance across all three metrics - accuracy, F1 score, and AUC, with respective scores of 0.813, 0.821, and 0.915. These findings demonstrate that the proposed method can be used in real-world clinical settings to diagnose Alzheimer's stages, especially MCI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

> **TODO: Add requirements.txt**

* spkit
  ```sh
  pip install spkit
  ```

### How to run

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install prerequisites
   ```sh
   pip install -r requirements.txt
   ```
3. Run main.py
   ```sh
   python main.py --gpu [GPU ID to use]
                  --mode [segmentation | extraction | selection | classification(default)]
                  --exp [1(default) - Tasks | 2 - Modals | 3 - Previous study]
                  --task [R(Resting) | C(Oddball) | N(1-back) | V(Verbal fluency)]
                  --seed [Random seed number]
                  --cv_num [Number of cross-validation folds]
                  --clf_type [Tree(default) | SVM | RF | MLP]
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- EXPERIMENTAL RESULTS -->
## Experimental Results

> Experiment 1. Evaluating the Contribution of Each Task in Alzheimer’s Classification

| <center>  </center> | <center> Resting </center> | <center> 1-back </center> | <center> Oddball </center> | <center> Verbal </center> | <center> Accuracy </center> | <center> F1 score </center> | <center> AUC score </center> |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|**Exp 1-A** | <center> O </center> | <center> O </center> | <center> O </center> | <center> O </center> | <center> 0.8126 </center> | <center> 0.8209 </center> | <center> 0.9149 </center> |
|**Exp 1-B** | <center> X </center> | <center> O </center> | <center> O </center> | <center> O </center> | <center> 0.8047 </center> | <center> 0.8095 </center> | <center> 0.9151 </center> |
|**Exp 1-C** | <center> O </center> | <center> X </center> | <center> O </center> | <center> O </center> | <center> 0.7845 </center> | <center> 0.7830 </center> | <center> 0.9030 </center> |
|**Exp 1-D** | <center> O </center> | <center> O </center> | <center> X </center> | <center> O </center> | <center> 0.7362 </center> | <center> 0.7440 </center> | <center> 0.8741 </center> |
|**Exp 1-E** | <center> O </center> | <center> O </center> | <center> O </center> | <center> X </center> | <center> 0.7357 </center> | <center> 0.7497 </center> | <center> 0.8798 </center> |


> Experiment 2. Evaluating the Contribution of Using Both EEG and fNIRS Signals in Alzheimer’s Classification

| <center>  </center> | <center> EEG </center> | <center> fNIRS </center> | <center> Accuracy </center> | <center> F1 score </center> | <center> AUC score </center> |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|**Exp 2-A** | <center> O </center> | <center> O </center> | <center> 0.8126 </center> | <center> 0.8209 </center> | <center> 0.9149 </center> |
|**Exp 2-B** | <center> O </center> | <center> X </center> | <center> 0.7079 </center> | <center> 0.7002 </center> | <center> 0.8847 </center> |
|**Exp 2-C** | <center> X </center> | <center> O </center> | <center> 0.6345 </center> | <center> 0.6336 </center> | <center> 0.7349 </center> |


> Experiment 3. Comparative Analysis with [Prior Research](https://www.sciencedirect.com/science/article/pii/S0165027020300406) Method

| <center>  </center> | <center> Accuracy </center> | <center> F1 score </center> | <center> AUC score </center> |
|:--------:|--------:|--------:|--------:| 
|**Exp 3-A** | <center> 0.8126 </center> | <center> 0.8209 </center> | <center> 0.9149 </center> |
|**Exp 3-B** | <center> 0.6855 </center> | <center> 0.6924 </center> | <center> 0.8359 </center> |
|**Exp 3-C** | <center> 0.6894 </center> | <center> 0.6753 </center> | <center> 0.8702 </center> |
|**Exp 3-D** | <center> 0.6975 </center> | <center> 0.6675 </center> | <center> 0.8777 </center> |
|**Exp 3-E** | <center> 0.5944 </center> | <center> 0.5944 </center> | <center> 0.6890 </center> |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Sunghyeon Kim - hahala25@yonsei.ac.kr

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Signal Processing toolkit (spkit)](https://github.com/Nikeshbajaj/spkit/tree/master)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

