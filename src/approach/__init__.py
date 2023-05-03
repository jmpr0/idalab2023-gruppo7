import os

# list all approaches available
__all__ = list(
    map(lambda x: x[:-3],
        filter(lambda x: x not in ['__init__.py', 'incremental_learning.py'] and x.endswith('.py'),
               os.listdir(os.path.dirname(__file__))
               )
        )
)

model_growth_apprs = ['chen2021', 'expert_gate', 'multiclass_classifiers','combiner']
ova_mg_approaches = ['chen2021']  # OneVsAll-ModelGrowth-Appraoches
