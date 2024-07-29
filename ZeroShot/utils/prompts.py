from typing import List, Tuple

def _get_default_text_promtps(category) -> Tuple[List, List]:
    pos_query = [
        f'There is {category.lower()}. ',
    ]
    neg_query = [
        "the lung is clear.",
        f'There is no {category.lower()}. ',
        f'no {category.lower()}. ',
        f"no evidence of {category.lower()}. ",
    ]
    return pos_query, neg_query

def _get_siim_text_promtps(category) -> Tuple[List, List]:
    """
    Custom text prompts for debugging only
    """
    Warning("Using debugging text prompts")
    pos_query = [
        "Abnormal collection of air in the pleural space between the lung and the chest wall. ",
        f'There is {category.lower()}. ',
        ]
    neg_query = [
        "the lung is clear.",
        f'There is no {category.lower()}. ',
        f'no {category.lower()}. ',
        f"no evidence of {category.lower()}. ",
        "The lung parenchyma is homogeneous, and the lung markings are normal and symmetrical. "
        ]
    return pos_query, neg_query

def get_all_text_prompts(categories, args):
    prompts_func_collection = {
        "default": _get_default_text_promtps,
        "siim": _get_siim_text_promtps,
    }

    assert args.text_prompt in prompts_func_collection.keys(), f"text_prompt {args.text_prompt} is not supported."
    prompts_func = prompts_func_collection[args.text_prompt]

    prompts = {}
    for cat in categories:
        pos_query, neg_query = prompts_func(cat)
        prompts[cat] = {
            "pos": pos_query,
            "neg": neg_query,
        }
    return prompts


def get_grounding_prompts(text_prompt):
    text = (
        'The ground-truth bounding box annotation for the phrase'
        f' *{text_prompt}* is shown in the middle figure (in black).'
    )
    return text