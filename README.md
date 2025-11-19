CMM3 – Quarter-Car Suspension Design (Python)
This codebase implements the required numerical methods (root finding, ODEs, regression/interpolation) to support a design decision for a quarter‑car suspension:

Tune suspension stiffness ks to meet a target sprung natural frequency band.
Simulate passive vs semi‑active skyhook damping under representative road inputs.
Compute comfort (RMS body acceleration), road‑holding (tyre deflection), and travel limits.
Structure
CMM3_Suspension_Project/
├── damping.py          # Passive & skyhook damper laws
├── design.py           # Root finding + natural frequency helpers
├── fit_damper.py       # Regression/interpolation of F–v damper data(demo)
├── io_utils.py         # I/O helpers
├── main.py             # Runs the full analysis and saves figures + summary
├── metrics.py          # RMS, peaks, and signal construction
├── model.py            # ODE RHS and RK4 solver
├── outputs/            # Created at runtime with PNGs + summary.json
├── params.py           # All parameters and constraints (edit here)
└── signals.py          # Road input generators (half-sine bump, sine sweep)
Setup for virtual environment
Create virtual environment
Activate virtual environment
Install python packages to virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
How to Run
python main.py
This will:

Compute ks to target a 1.45 Hz sprung frequency (midpoint of 1.3–1.6 Hz band).
Simulate passive damping over a 50 mm half‑sine bump at 10, 20, 30 m/s.
Simulate semi‑active skyhook for the 20 m/s case.
Save figures and a summary.json in outputs/.
Demonstrate regression/interpolation by fitting a noisy F–v damper dataset.
Design Metrics
Comfort: RMS of body acceleration (xsddot).
Road holding: RMS of tyre deflection (xu - y) and peak.
Travel: Peak of suspension travel (xs - xu).
Compare metrics to constraints defined in params.py:

Travel limit: 75 mm
Tyre deflection limit: 15 mm
Notes
The ODE is solved with a fixed‑step RK4 (model.rk4), which is adequate for this system.
Skyhook control uses an on‑off logic: high damping when ẋs and relative velocity ẋs−ẋu have the same sign.
design.coupled_natural_frequencies reports the two coupled undamped modes for sanity checks against target bands.
Extensions (suggested in the report)
Use frequency weighting for comfort (ISO 2631‑1) instead of unweighted RMS.
Replace the on‑off skyhook with a continuous law or clipped‑optimal control.
Include tyre damping and/or a more detailed tyre model.
Swap to an adaptive ODE solver and step‑size control.
Add parameter sweeps and multi‑objective trade‑off plots.