# Table I - DF-R
#python src/util/experiment.py --dataset lm --mode 0                                 # baseline, 2 iterations/obj
#python src/util/experiment.py --dataset lm --mode 1                                 # PIR
#python src/util/experiment.py --dataset lm --mode 2                                 # SIR
python src/util/experiment.py --dataset lm --mode 0 --refinement_iterations 10      # baseline, 10 iterations/obj
python src/util/experiment.py --dataset lm --mode 3                                 # RIR

# Table I - ICP (requires the PCL bindings)
#python src/util/experiment.py --dataset lm --mode 0 --lm_baseline ppf               # baseline, 5*10 iterations/obj
#python src/util/experiment.py --dataset lm --mode 1 --lm_baseline ppf               # PIR
#python src/util/experiment.py --dataset lm --mode 2 --lm_baseline ppf               # SIR
python src/util/experiment.py --dataset lm --mode 0 --lm_baseline ppf --refinement_iterations 15  # baseline, 15*10 iterations/obj
python src/util/experiment.py --dataset lm --mode 3 --lm_baseline ppf               # RIR

# Table II - TrICP (requires the PCL bindings)
#python src/util/experiment.py --dataset xapc --mode 3                               # RIR
#python src/util/experiment.py --dataset xapc --mode 4                               # VFb
python src/util/experiment.py --dataset xapc --mode 5                               # VFd

# Table III - DF-R
#python src/util/experiment.py --dataset ycbv --mode 0                               # baseline, 2 iterations/obj
#python src/util/experiment.py --dataset ycbv --mode 1                               # PIR
#python src/util/experiment.py --dataset ycbv --mode 2                               # SIR
python src/util/experiment.py --dataset ycbv --mode 0 --refinement_iterations 10    # baseline, 10 iterations/obj
#python src/util/experiment.py --dataset ycbv --mode 3                               # RIR
#python src/util/experiment.py --dataset ycbv --mode 4                               # VFb
python src/util/experiment.py --dataset ycbv --mode 5                               # VFd