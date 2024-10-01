# Welcome to the PyWASPGEN Python Package!

<p align="center">
  <img src="https://github.com/user-attachments/assets/ba1642ed-5193-4b99-bf99-048c6f5cec9c" width="600px"/>
</p>

PyWASPGEN (Python Wideband Aggregate SPectrum GENerator) is intended as a native python dataset generation tool for creating synthetic aggregate radio frequency captures for initial testing and evaluation of spectrum sensing algorithms. The data produced by this tool is particularly useful for testing signal detection algorithms (i.e. where in time and frequency signals exist in the capture) as well as signal classification algorithms (i.e. what is the signaling format of the detected signal).

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PyWASPGEN from the root directory of the repository.

```bash
pip install .
```

## For Developers
If you're interested in contributing to the development of PyWASPGEN, you'll need to install `pre-commit`.

```bash
pip install pre-commit
pre-commit install
```

## Usage
Generating synthetic radio frequency captures using PyWASPGEN can either be done directly through user-specified signal generation parameters or pseudorandomly through user-specified signal generation parameter ranges.

### Direct Capture Generation (see example script below for detailed comments)
```bash
python examples/direct_generation.py
```

### Pseudorandom Capture Generation (see example script below for detailed comments)
```bash
python examples/random_generation.py
```
## Acknowledgements
PyWASPGEN is based upon work supported, in whole or in part, by the U.S. Department of Defense through the Office of the Assistant Secretary of Defense for Research and Engineering (ASD(R&E)) under Contract HQ003419D0003. The Systems Engineering Research Center (SERC) is a federally funded University Affiliated Research Center managed by Stevens Institute of Technology. Any views, opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Department of Defense nor ASD(R&E).

## Contributors
| Name | Role | Title | Email |
| ---- | ---- | ----- | ----- |
| William 'Chris' Headley | Developer | Associate Director, Spectrum Dominance Division, Virginia Tech National Security Institute | cheadley@vt.edu |
| Caleb McIrvin | Developer | PhD Student, Spectrum Dominance Division, Virginia Tech National Security Institute | calebmcirvin111@vt.edu |
| Michael 'Alex' Kyer | Developer | Software Engineer, Intelligent Systems Division, Virginia Tech National Security Institute | makyer19@vt.edu |

## License
[MIT](https://choosealicense.com/licenses/mit/)
