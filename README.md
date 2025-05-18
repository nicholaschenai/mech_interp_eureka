# Mech Interp for Robots -- On a Franka Robot Arm Model trained with Eureka

A project focused on mechanistic interpretability of neural networks trained with automatically discovered reward functions through the Eureka algorithm. The project specifically analyzes a Franka robot arm model trained to open a drawer, investigating how the model's internal mechanisms work and how they relate to the automatically generated reward functions.

The report can be found [here](./docs/final_report.md)

## Folder structure
```
/
├── ckpts/                        # Model checkpoints and saved states
├── config/                       
├── docs/                         
├── notebooks/                    # Jupyter notebooks
│   ├── eda/                      # Basic exploration of models and environment
│   └── mech_interp/              # Mechanistic interpretability analyses
├── results/                      # Generated analysis results and visualizations
├── scripts/                      
│   ├── correlation/              # Correlation analysis scripts
│   ├── circuits/                 # Circuit analysis scripts
│   └── sae/                      # Sparse Autoencoder analysis scripts
├── src/                          
├── tests/                        
├── utils/                        
├── README.md                     
└── requirements.txt              # Dependencies (TODO)
```

## Getting Started
To train models via Eureka, follow the instructions [here](https://github.com/nicholaschenai/eureka_exploration) to do so. This is quite intensive so we included the trained models in the `ckpts/` folder for convenience.

1. Clone the repository
2. Install dependencies (TODO, use the one from Eureka above for now as it is a superset):
   ```bash
   pip install -r requirements.txt
   ```
3. Follow the notebooks in `notebooks/` for initial exploration and `scripts/` for standalone scripts
4. Check `docs/` for detailed documentation and analysis reports especially the final report [here](./docs/final_report.md)
