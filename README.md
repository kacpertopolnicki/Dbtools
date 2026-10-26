# Dbtools

Tools for hamiltonian monte carlo and gaussian processes.

## Structure

```
├── configs           
│   │   ################################
│   │   #  SAMPLE CONFIGURATION FILES  #
│   │   ################################
│   │   
│   │   #    - ..._gp are for gaussian processes 
│   │   #    - ..._hmc are for hamiltonian monte carlo
│   ├── config1_gp                  
│   ├── config1_hmc
│   ├── config2_hmc
│   ├── config_gp
│   └── config_hmc
├── examples
│   │   ##############
│   │   #  EXAMPLES  #
│   │   ##############
│   │   
│   │   #  Sample usage: `./example1HMC.jl ../configs/config1_hmc temp_1dchain_21-10-2026` 
│   ├── example1HMC.jl
│   │   #  Sample usage: `./example2HMC.jl ../configs/config1_hmc ../configs/config2_hmc ../configs/config1_gp temp_1dchain_26-10-2026`
│   ├── example2HMC.jl
│   │   #  Sample usage: `./exampleGP.jl ../configs/config_gp temp.png`
│   ├── exampleGP.jl
│   │   #  Sample usage: `./exampleHMC.jl ../configs/config_hmc temp_chain_17-10-2026_999`
│   ├── exampleHMC.jl
│   │   #  For plotting results of ...eHMC.jl, ... e1HMC.jl
│   ├── plot1_hmc.nb
│   │   #  For plotting results of ...eHMC2.jl
│   └── plot_hmc.nb
├── LICENSE
├── Manifest.toml
├── Project.toml
├── README.md
└── src
    │   ##########
    │   #  CODE  #
    │   ##########
    │   
    ├── Dbtools.jl
    ├── GaussianProcesses.jl
    └── HamiltonianMonteCarlo.jl
```
