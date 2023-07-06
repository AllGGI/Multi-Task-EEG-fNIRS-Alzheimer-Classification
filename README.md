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
  <li><a href="#usage">Usage</a></li>
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

This is an example of ~.

### Prerequisites

TODO: Add requirements.txt

* spkit
  ```sh
  pip install spkit
  ```

### How to run

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install prerequisites
   ```sh
   pip install -r requirements.txt
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- EXPERIMENTAL RESULTS -->
## Experimental Results

> Experiment 1. Evaluating the Contribution of Each Task in Alzheimer’s Classification

| <center>  </center> | <center> Accuracy </center> | <center> F1 score </center> | <center> AUC score </center> |
|:--------:|--------:|--------:|--------:| 
|**Exp 1-A** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 1-B** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 1-C** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 1-D** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 1-E** | <center> % </center> | <center> % </center> | <center> % </center> |


> Experiment 2. Evaluating the Contribution of Using Both EEG and fNIRS Signals in Alzheimer’s Classification

| <center>  </center> | <center> Accuracy </center> | <center> F1 score </center> | <center> AUC score </center> |
|:--------:|--------:|--------:|--------:| 
|**Exp 2-A** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 2-B** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 2-C** | <center> % </center> | <center> % </center> | <center> % </center> |


> Experiment 3. Comparative Analysis with [Prior Research](https://www.sciencedirect.com/science/article/pii/S0165027020300406) Method


| <center>  </center> | <center> Accuracy </center> | <center> F1 score </center> | <center> AUC score </center> |
|:--------:|--------:|--------:|--------:| 
|**Exp 3-A** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 3-B** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 3-C** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 3-D** | <center> % </center> | <center> % </center> | <center> % </center> |
|**Exp 3-E** | <center> % </center> | <center> % </center> | <center> % </center> |



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Sunghyeon Kim - hahala25@kaist.ac.kr

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

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

