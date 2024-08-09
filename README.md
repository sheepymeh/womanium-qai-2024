# QML-for-Conspicuity-Detection-in-Production
Womanium Quantum+AI 2024 Projects

**Please review the participation guidelines [here](https://github.com/womanium-quantum/Quantum-AI-2024) before starting the project.**

_**Do NOT delete/ edit the format of this read.me file.**_

_**Include all necessary information only as per the given format.**_

## Project Information:

### Team Size:
  - Maximum team size = 2
  - While individual participation is also welcome, we highly recommend team participation :)

### Eligibility:
  - All nationalities, genders, and age groups are welcome to participate in the projects.
  - All team participants must be enrolled in Womanium Quantum+AI 2024.
  - Everyone is eligible to participate in this project and win Womanium grants.
  - All successful project submissions earn the Womanium Project Certificate.
  - Best participants win Womanium QSL fellowships with Fraunhofer ITWM. Please review the eligibility criteria for QSL fellowships in the project description below.

### Project Description:
  - Click [here](https://drive.google.com/file/d/1AcctFeXjchtEhYzPUsHpP_b4HGlI4kq9/view?usp=sharing) to view the project description.
  - YouTube recording of the project description - [link](https://youtu.be/Ac1ihFcTRTc?si=i6AIVfQQh8ymYQYp)

## Project Submission:
All information in this section will be considered for project submission and judging.

Ensure your repository is public and submitted by **August 9, 2024, 23:59pm US ET**.

Ensure your repository does not contain any personal or team tokens/access information to access backends. Ensure your repository does not contain any third-party intellectual property (logos, company names, copied literature, or code). Any resources used must be open source or appropriately referenced.

### Team Information:
Team Member 1:
 - Full Name: Lim See Min
 - Womanium Program Enrollment ID: `WQ24-ZqoNYjuEvlcpS25`


Team Member 2:
 - Full Name: Wu Jiayang
 - Womanium Program Enrollment ID: `WQ24-E9wFvjuPiAYTHLg`


### Project Solution:

#### Task 1

We completed the "Introduction to Quantum Computing", "Single-Qubit Gates", and "Circuits with Many Qubits" sections of the Pennylane codebook and present our original solutions, summaries and explanations about them.

#### Task 2

We implemented a variational classifier for the Iris dataset. To further our understanding, we extend the tutorial by classifying between 3 types of flowers. This required one-hot encoding as it is no longer a binary classification problem. We also utilize 4 features to enhance the model's abilities to classify.

#### Task 3

We built a Quanvolutional Neural Network. To improve performance, we allow the QNN parameters to be trained through gradients calculated by JAX. We used this task as a starting point to learn to write JAX+Pennylane code and improve performance in the simulator by using `jax.vmap`.

#### Task 4

We first implemented a sine function model based on a [Pennylane tutorial](https://pennylane.ai/blog/2021/10/how-to-start-learning-quantum-machine-learning/) and evaluated this, showing why it is too restrictive. We then create our quantum architecture, discussing how our model has more degrees of freedom. We finish by training our model on a summed sine and cosine function and discussing the implications in model periodic functions based on Fourier series expansions.

#### Task 5

We combine a randomly initialized ResNet-18 model with the Quanvolutional Neural Network from Task 3 sequentially. We also implemented simple image augmentations and dataset balancing to minimize overfitting.


### Project Presentation Deck:
[The slide deck is available here on Google Slides](https://docs.google.com/presentation/d/1qUByHxJ93iQsU6faiowyLAIzHhBio2wGTxkthyWygVE/preview)

