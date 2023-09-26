# inHERA-msc-thesis
inHERA-msc-thesis is a Python library to test repertoire-based robotic adaptation algorithms with hardware acceleration support.
It also proposes a new class of algorithms called Experience-Sharing Adaptation Algorithms.

## Description
This repository contains the adaptation pipeline written during Matteo G. Mecattaf's
master's project for the MSc in Artificial Intelligence delivered by Imperial College London.

This project enables repertoire-based robotic adaptation to be tested using hardware acceleration tools.

For more details, see the associated master's thesis.

## Visuals
The following are some examples of the visual results that can be produced with this library.
### Dynamic [QDax](https://qdax.readthedocs.io/en/latest/) grid during ITE adaptation

#### Mean
<div align="center">
<img alt="mean" height="350" width="350" src="/docs/media/ite_mean_adaptation.gif"/>
</div>

#### Variance
<div align="center">
<img alt="variance" height="350" width="350" src="/docs/media/ite_variance_adaptation.gif"/>
</div>

### Adaptation plots
<div align="center">
<img alt="adaptation" height="230" src="/docs/media/adaptation_plots.png"/>
</div>

## Installation
For installation, please refer to the [AIRL wiki](https://gitlab.doc.ic.ac.uk/AIRL/AIRL_WIKI) for tips on using Singularity.

## Usage
The main usages of this library are:
- To generate MAP-Elites grids (see [repertoire_creation](src_mains/repertoire_creation) folder)
- To create robotic ancestors (see [ancestors_generation.py](src_mains/agent_creation/ancestors_generation.py) file)
- To create robotic children (see [children_generation.py](src_mains/agent_creation/children_generation.py))
- To visualise results (see [visualisations](src_mains/visualisations) folder)

Note that all the example uses listed above and more can be found in the [src_mains](src_mains) folder.
All the example main files in this folder use the classes written in the [src](src) folder.

## Support
For any questions related to this project, feel free to message me on LinkedIn: [Matteo G. Mecattaf](https://www.linkedin.com/in/matteo-mecattaf/).

## Acknowledgment
Thank you to the supervisor of this thesis Dr Antoine Cully who has offered invaluable help throughout the project.

## License
[MIT](https://choosealicense.com/licenses/mit/)



