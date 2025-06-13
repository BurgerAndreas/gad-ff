"""
Train a ML FF to predict the GAD vector field.

Caveat: NewtonNet predicts conservative force, i.e. by taking the negative gradient of the potential energy w.r.t. the atomic positions.
There is not potential energy of the GAD vector field.
Instead we want to directly predict the GAD vector field.

Possible approaches:
- start from augmented-T1x-finetuned NewtonNet model (https://www.nature.com/articles/s41467-024-52481-5), train randomly-initialized force prediction head, keep backbone frozen
- start from augmented-T1x-finetuned NewtonNet model (https://www.nature.com/articles/s41467-024-52481-5), train randomly-initialized force prediction head and backbone
- start from randomly-initialized NewtonNet model, train force prediction head and backbone from scratch
"""
