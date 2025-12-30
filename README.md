# Nonlinear Crystal SHG Calculator v2.0

## 🚀 v2.0 Release Notes / 更新日志
We are excited to announce **v2.0**! This update extends our simulation capabilities to Sum Frequency Generation and introduces temperature tuning mechanics.

-   **SFG Support**: Now supports nonlinear parameter calculations for Sum Frequency Generation (SFG).
-   **Temperature Matching**: Added a new module for Temperature Phase Matching (TPM).
-   **Enhanced Sellmeier Library**: Users can now select Sellmeier equations from different references. We will continue to expand this database in future updates.

## Introduction
This is an optical simulation tool developed using **Python** and **Streamlit**. It is designed to calculate critical parameters for frequency shifthing processes in common nonlinear crystals, such as BBO, LBO, CLBO, and KDP.

## Features
- **Multi-Crystal Support**: Built-in database containing parameters for common crystals including BBO, LBO, CLBO, KDP, and DKDP.
- **Phase Matching Calculation**: Supports precise calculation of phase matching angles for both **Type I** and **Type II** interactions.
- **Effective Nonlinear Coefficient ($d_{eff}$)**: Calculates $d_{eff}$ for various crystal orientations based on **IEEE standards** and **Kleinman symmetry**.
- **Interactive Interface**: Provides a user-friendly web interface for real-time parameter tuning and result visualization.
- **3D Visualization**: Supports 3D rendering of crystal orientation schematics and refractive index ellipsoids.

## Demo
<img width="1942" height="761" alt="image" src="https://github.com/user-attachments/assets/9aa36358-daf4-4f6e-b337-79a3378dbba9" />
<img width="1370" height="911" alt="image" src="https://github.com/user-attachments/assets/935ee456-5ed0-40ae-994e-2962b8b17c7b" />
<img width="1602" height="851" alt="image" src="https://github.com/user-attachments/assets/db562468-bf57-4e8a-a86b-7a42279e0245" />


## Installation & Usage

1. **Clone the repository**
   
   git clone [https://github.com/Hongxin-Chen/Nonlinear_Crystal_Simulation_of_SHG.git](https://github.com/Hongxin-Chen/Nonlinear_Crystal_Simulation_of_SHG.git)
   cd Nonlinear_Crystal_Simulation_of_SHG

2. **Install dependencies**
   
   pip install -r requirements.txt

3. **Run the application**

   streamlit run launcher.py
