import glob
import math
import os.path
import random
import re
import string
from functools import partial
from typing import Callable, List

import pandas as pd
import yaml
from utils import load_yaml, str_presenter

DEFAULT_SIGNIFICANT_DIGITS = 3

standard_tabular_text_templates = [
    "The molecule with the {SMILES__description} representation of {SMILES#} has a {TARGET__names__noun} of {TARGET#}.",  # noqa: E501
    "The molecule {SMILES#} has a {TARGET__names__noun} of {TARGET#}.",
    "Based on the {SMILES__description} representation {SMILES#}, the molecule has a {TARGET__names__noun} of {TARGET#}.",  # noqa: E501
    "Based on the representation {SMILES#}, the molecule has a {TARGET__names__noun} of {TARGET#}.",  # noqa: E501
    "The {SMILES__description} {SMILES#} represents a molecule that has a {TARGET__names__noun} of {TARGET#}.",  # noqa: E501
    "The {SMILES__description} {SMILES#} represents a molecule with a {TARGET__names__noun} of {TARGET#}.",  # noqa: E501
    "{SMILES#} represents a molecule that has a {TARGET__names__noun} of {TARGET#}.",  # noqa: E501
    "{SMILES#} represents a molecule with a {TARGET__names__noun} of {TARGET#}.",  # noqa: E501
    "The {SMILES__description} {SMILES#} has a {TARGET__names__noun} of {TARGET#}.",
    "The molecule {SMILES#} has a {TARGET__names__noun} of {TARGET#}.",
    "{SMILES#} has a {TARGET__names__noun} of {TARGET#}.",
    "The {TARGET__names__noun} of the {SMILES__description} {SMILES#} is:<EOI> {TARGET#}",  # noqa: E501
    "The {TARGET__names__noun} of the {SMILES__description} {SMILES#} is<EOI> {TARGET#}.",  # noqa: E501
]


exclude_from_standard_tabular_text_templates = [
    "ames_mutagenicity",  # because it is boolean target data
    "bioavailability_ma_et_al",  # because it is boolean target data
    "blood_brain_barrier_martins_et_al",  # because it is boolean target data
    "chembl_v29",  # text only, no SMILES
    "moses",  # SMILES only, has no target
    "zinc",  # SMILES only, has no target
]


lm_eval_yaml_template_loglikelihood = {
    "group": [
        "loglikelihood",
    ],
    "task": None,
    "dataset_path": None,
    "dataset_name": None,
    "output_type": "loglikelihood",
    "test_split": "test",
    "template_aliases": "",
    "doc_to_text": "{{input}}",
    "doc_to_target": "{{output}}",
    # "should_decontaminate": True,
    # "doc_to_decontamination_query": "{{text}}",
    "metric_list": [
        {
            "metric": "perplexity",
            "aggregation": "perplexity",
            "higher_is_better": False,
        },
        {
            "metric": "acc",
            "aggregation": "mean",
            "higher_is_better": True,
        },
    ],
}

lm_eval_yaml_template_multiple_choice = {
    "group": [
        "multiple_choice",
    ],
    "task": None,
    "dataset_path": None,
    "dataset_name": None,
    "output_type": "multiple_choice",
    "test_split": "test",
    "template_aliases": "{% set gold = correct_output_index %}",
    "doc_to_text": "{{input}}",
    "doc_to_target": "{{output}}",
    "gold_alias": "{{gold}}",
    # "should_decontaminate": True,
    # "doc_to_decontamination_query": "{{text}}",
    "metric_list": [
        {
            "metric": "acc",
            "aggregation": "mean",
            "higher_is_better": True,
        },
        {
            "metric": "acc_norm",
            "aggregation": "mean",
            "higher_is_better": True,
        },
        # todo: check acc_mutual_info because it breaks
        # {
        #     "metric": "acc_mutual_info",
        #     "aggregation": "mean",
        #     "higher_is_better": True,
        # },
    ],
}


def unwrap_list_length_1(list_input: list):
    """Unwraps lists of length 1 and returns the first = single element."""
    if isinstance(list_input, list):
        assert len(list_input) == 1
        return list_input[0]
    else:
        raise NotImplementedError()


class RandomVariable:
    """Simple random variable class that takes in a name, data, and a sampler.
    The sampler needs to return a single element."""

    def __init__(self, name: str, data: list, sampler: Callable = None):
        self.name = name
        self.data = data
        self.sampler = partial(random.sample, k=1) if sampler is None else sampler

    def __repr__(self):
        return f"RandomVariable: {self.name}, {self.data}, {self.sampler}"

    def __call__(self) -> str:
        """Carries out sampling and returns a single element."""
        return unwrap_list_length_1(self.sampler(self.data))


def get_input_variables_from_template(template: str) -> List[str]:
    """Gets all strings that are between curly brackets that are used downstream as input variables."""
    return [x for x in re.findall(r"\{([^}]+)\}", template)]


class PromptTemplate:
    """Simple prompt template class that takes a string text template as input.
    The input_variables can be defined but if not they are derived from the text template (= default use case).
    """

    def __init__(self, template: str, input_variables: List[str] = None):
        self.template = template
        if input_variables is None:
            self.input_variables = self.get_input_variables()

    def get_input_variables(self) -> List[str]:
        """Gets all variable strings from the text template that are between curly brackets
        that are used as input_variables."""
        return get_input_variables_from_template(self.template)

    def __repr__(self):
        return f"PromptTemplate: {self.template}"

    def insert(self, data: dict) -> str:
        """Inserts the data and checks before if we got all the data for the present input_variables.
        More input_variables in data doesn't matter."""
        assert all(x in data.keys() for x in self.input_variables)
        template = self.template
        for k in data:
            if k not in self.input_variables:
                continue
            template = template.replace("{" + k + "}", data[k])
        return template


def get_random_text_identifiers_and_targets(meta: dict) -> dict:
    """Gets random variables as RandomVariable objects from the identifiers and targets
    in the meta dict derived from the meta.yaml."""
    rnd_texts = {}
    for e in meta["identifiers"] + meta["targets"]:
        rnd_texts[e["id"]] = {}
        if "names" in e:
            rnd_texts[e["id"]]["names"] = {}
            name_types = set([list(x.keys())[0] for x in e["names"]])
            for name in name_types:
                rnd_text = RandomVariable(
                    f"{e['id']}__names__{name}",
                    [x[name] for x in e["names"] if name in x],
                )
                rnd_texts[e["id"]]["names"][name] = rnd_text
        else:
            rnd_texts[e["id"]]["description"] = partial(
                lambda x: x, e["description"]
            )  # to wrap value in function = deterministic, no sampling

    return rnd_texts


def get_target_from_string(meta: dict, string: str) -> str:
    """Gets a target string from the meta dict based on the variable string.
    (A variable string is what is between the curly brackets in the text template.)"""
    keys = string.split("__")

    def get_with_nested_keys(d: dict, keys: list) -> str:
        t = d.copy()
        for k in keys:
            t = t[k]
        return t

    if len(keys) == 1 and keys in meta:
        return meta[
            keys
        ]  # assumes single element in meta and doesn't not support nested dicts in that version
    elif keys[0] in [x["id"] for x in meta["identifiers"] + meta["targets"]]:
        rnd_texts = get_random_text_identifiers_and_targets(meta)
        return get_with_nested_keys(rnd_texts, keys)
    else:
        raise NotImplementedError()


def get_symbols_from_multiple_choice_enum(
    multiple_choice_enum: str,
) -> List[str]:
    """Create from the multiple_choice_enum variable a string of the symbols
    that are used as multiple choice enumeration symbols.

    Example:
    %multiple_choice_enum%3-5%aA1

    %multiple_choice_enum ... id
    %3-5 ... multiple choice count
    %aA1 ... symbol definition
    """

    multiple_choice_enum_split = multiple_choice_enum[1:].split("%")
    assert (
        len(multiple_choice_enum_split) == 3
    ), "Wrong multiple_choice_enum field setup."
    _, choice_count, symbol = multiple_choice_enum_split

    # get the choice_count
    if len(choice_count) >= 3:
        assert (
            "-" in choice_count
        ), "The choice count needs to consist of two integers separated by a `-`."
        min_, max_ = [int(x) for x in choice_count.split("-")]
        assert isinstance(min_, int) and isinstance(
            max_, int
        ), "The choice count needs to consist of two integers."
        choice_count_sampled = random.randint(min_, max_)
    elif len(choice_count) == 1:
        choice_count_sampled = int(choice_count)
    else:
        raise NotImplementedError()

    # get the symbols
    assert any(
        [x in symbol for x in "aA1"]
    ), "Allowed symbols are `a` (lower case letters), `A` (upper case letters), and/or `1` (integers)."
    symbol_sampled = random.sample(symbol, k=1)[0]
    if symbol_sampled == "a":
        symbols = list(string.ascii_lowercase[:choice_count_sampled])
    elif symbol_sampled == "A":
        symbols = list(string.ascii_uppercase[:choice_count_sampled])
    elif symbol_sampled == "1":
        symbols = [str(x + 1) for x in range(choice_count_sampled)]

    return symbols


class TemplateSampler:
    """The template sampler uses the data_clean.csv and meta.yaml from a data directory and
    manages the the insertion of the sampled data into the text templates."""

    def __init__(
        self,
        path_data_dir: str,
        path_lm_eval_data_dir: str,
        multiple_choice_rnd_symbols: list,  # = ["", ".", ".)", ")", ":", "()", "[]"],
        additional_templates: list = None,
        template_sampler: Callable = None,
        column_datafield_sampler: Callable = None,
        benchmarking_templates: bool = False,
        multiple_choice_benchmarking_templates: bool = False,
        multiple_choice_benchmarking_format: int = 0,
    ):
        # paths
        self.path_data_dir = path_data_dir
        self.path_data_meta = self.path_data_dir + "/meta.yaml"
        self.path_data_csv = self.path_data_dir + "/data_clean.csv"
        self.path_lm_eval_data_dir = path_lm_eval_data_dir

        # meta from yaml
        self.meta = load_yaml(self.path_data_meta)

        # dataframe from csv
        df = pd.read_csv(self.path_data_csv)

        def check_targets_and_identifiers(meta: dict, df: pd.DataFrame):
            all_identifiers = [x["id"] for x in meta["identifiers"]] + [
                x["id"] for x in meta["targets"]
            ]
            all_identifiers
            for i in all_identifiers:
                cols = df.columns.tolist()
                assert i in cols, f"target or identifier {i} not in columns {cols}!"

        check_targets_and_identifiers(self.meta, df)

        additional_targets = {
            "selfies": {
                "id": "selfies",
                "type": "seflies",
                "description": "SELFIES",
            },
            "deepsmiles": {
                "id": "deepsmiles",
                "type": "deepsmiles",
                "description": "DeepSMILES",
            },
            "canonical": {
                "id": "canonical",
                "type": "canonical",
                "description": "canonical SMILES",
            },
            "inchi": {
                "id": "inchi",
                "type": "inchi",
                "description": "InChI",
            },
            "tucan": {
                "id": "tucan",
                "type": "tucan",
                "description": "TUCAN",
            },
            "iupac_name": {
                "id": "iupac_name",
                "type": "iupac_name",
                "description": "IUPAC name",
            },
        }
        self.additional_targets = []
        for col in [
            "selfies",
            "deepsmiles",
            "canonical",
            "inchi",
            "tucan",
            "iupac_name",
        ]:
            if col in df.columns:
                self.additional_targets.append(col)
                self.meta["targets"].append(additional_targets[col])

        # assert not df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        if "split" not in df.columns:
            df["split"] = "train"
        self.df = df

        # text templates
        self.benchmarking_templates = benchmarking_templates
        self.multiple_choice_rnd_symbols = multiple_choice_rnd_symbols
        self.multiple_choice_benchmarking_templates = (
            multiple_choice_benchmarking_templates
        )
        self.multiple_choice_benchmarking_format = multiple_choice_benchmarking_format

        templates = self.meta.get("templates", [])
        if additional_templates:
            self.additional_templates = additional_templates
            templates += additional_templates

        if self.benchmarking_templates:
            templates = [t for t in templates if t.find("<EOI>") != -1]

            if self.multiple_choice_benchmarking_templates:
                templates = [t for t in templates if t.find("%multiple_choice_") != -1]
            else:
                templates = [t for t in templates if t.find("%multiple_choice_") == -1]
        else:
            templates = [t for t in templates if t.find("<EOI>") == -1]

        self.templates = templates
        print(f"\n### templates\n{self.templates}")
        assert self.templates is not None
        assert self.templates is not []
        self.prompt_templates = [PromptTemplate(t) for t in self.templates]

        # create random variables for prompts and texts
        self.rnd_prompt_templates = RandomVariable(
            "rnd_prompt_templates", self.prompt_templates, template_sampler
        )
        self.rnd_texts = get_random_text_identifiers_and_targets(self.meta)

        # column_datafield_sampler
        self.column_datafield_sampler = (
            partial(random.sample, k=1)
            if column_datafield_sampler is None
            else column_datafield_sampler
        )

    def __repr__(self):
        return f"TemplateSampler: {self.path_data_dir}"

    def _get_target_from_row(self, sample: pd.Series, var: str) -> str:
        """Get target string from sample row and variable string."""
        # sampling based on columns and their definiton in the text template
        if ("#" in var) and ("&" in var):  # recoding information in var
            var, choices = var.split("#")
            choices = choices.split("&")
            choice = choices[sample[var]]
            if choice == "NULL":
                out = ""
            else:
                out = choices[sample[var]]
        elif ("#" in var) and ("|" in var):  # use data from multiple columns
            columns = var.split("|")
            columns = [var.replace("#", "") for var in columns]
            choices = sample[columns].tolist()
            choices = [c for c in choices if (isinstance(c, str) or not math.isnan(c))]
            out = unwrap_list_length_1(self.column_datafield_sampler(choices))
        elif "#" in var:  # use only data from column
            out = sample[var.replace("#", "")]
            # for KG: if *_smiles is nan sample from *_name
            if (
                not isinstance(out, str)
                and math.isnan(out)
                and var.find("_smiles") != -1
            ):
                out = sample[var.replace("_smiles", "_name").replace("#", "")]
            # for KG: if *_protein_names is nan sample from *_name
            elif (
                not isinstance(out, str)
                and math.isnan(out)
                and var.find("_protein_names") != -1
            ):
                out = sample[var.replace("_protein_names", "_name").replace("#", "")]

        var_dict = [
            x
            for x in self.meta["identifiers"] + self.meta["targets"]
            if x["id"] == var.replace("#", "")
        ][0]
        data_type = var_dict["type"]
        if data_type == "continuous":
            assert isinstance(out, float)
            significant_digits = var_dict.get(
                "significant_digits", DEFAULT_SIGNIFICANT_DIGITS
            )
            out = str(f"{round(out, significant_digits):.{significant_digits}f}")
        else:
            out = str(out)

        # sampling based on row data and their definiton in the row
        if "|" in out:  # datafield sampling of multiple options
            choices = out.split("|")
            choices = [c for c in choices if (isinstance(c, str) or not math.isnan(c))]
            out = unwrap_list_length_1(self.column_datafield_sampler(choices))
        return out

    def get_sample_dict(self, sample: pd.Series, template: str):
        """Get sample dict from sample row and template string."""
        input_variables = get_input_variables_from_template(template)
        sample_dict = {}

        # multiple choice template setup
        if any([x.find("%") != -1 for x in input_variables]):
            # get multiple_choice_enum
            multiple_choice_enum_idx = [
                i
                for i, x in enumerate(input_variables)
                if x.startswith("%multiple_choice_enum")
            ]
            assert len(multiple_choice_enum_idx) == 1
            multiple_choice_enum_idx = multiple_choice_enum_idx[0]  # unpack list
            multiple_choice_enum = input_variables[multiple_choice_enum_idx]

            # get multiple_choice_var
            multiple_choice_var_idx = [
                i for i, x in enumerate(input_variables) if x.endswith("%")
            ]
            assert len(multiple_choice_var_idx) == 1
            multiple_choice_var_idx = multiple_choice_var_idx[0]  # unpack list
            multiple_choice_input = input_variables[multiple_choice_var_idx]
            if multiple_choice_input.count("%") > 1:
                (
                    multiple_choice_var,
                    multiple_choice_indicator,
                    _,
                ) = multiple_choice_input.split("%")
            else:
                (
                    multiple_choice_var,
                    multiple_choice_indicator,
                ) = multiple_choice_input.split("%")
                # multiple_choice_indicator is here a empty string

            symbols = get_symbols_from_multiple_choice_enum(multiple_choice_enum)

            # remove multiple choice control sequences from input_variables if present
            input_variables.remove(multiple_choice_enum)
            if multiple_choice_indicator == "":
                input_variables.remove(multiple_choice_var + "%")
            else:
                input_variables.remove(
                    multiple_choice_var + "%" + multiple_choice_indicator + "%"
                )
            input_variables.remove("%multiple_choice_result")

            # get all and correct choices incl. index
            correct_choice = self._get_target_from_row(
                sample, multiple_choice_var + "#"
            )

            if multiple_choice_indicator == "":
                # standard sampling w/o paired data
                all_choices = sorted(
                    [str(x) for x in self.df[multiple_choice_var].unique().tolist()]
                )
                if all_choices == ["0", "1"]:
                    all_choices = ["False", "True"]
                    correct_choice = all_choices[int(correct_choice)]
                multiple_choices = random.sample(all_choices, k=len(symbols))
                if correct_choice not in multiple_choices:
                    multiple_choices = multiple_choices[:-1] + [correct_choice]
                    random.shuffle(multiple_choices)
                correct_choice_idx = multiple_choices.index(correct_choice)
            else:
                # standard sampling w/ paired data and potentially multiple correct answers
                correct_choice_indicator = self._get_target_from_row(
                    sample, multiple_choice_indicator + "#"
                )
                df_sample = self.df.sample(len(symbols) - 1)[
                    [multiple_choice_var, multiple_choice_indicator]
                ]
                multiple_choices = df_sample[multiple_choice_var].astype(str).tolist()
                multiple_choices_indicators = (
                    df_sample[multiple_choice_indicator].astype(str).tolist()
                )
                del df_sample

                multiple_choices += [correct_choice]
                multiple_choices_indicators += [correct_choice_indicator]
                multiple_choices_combined = list(
                    zip(multiple_choices, multiple_choices_indicators)
                )  # create list of tuples to keep track of indicators
                random.shuffle(multiple_choices_combined)
                multiple_choices, multiple_choices_indicators = list(
                    zip(*multiple_choices_combined)
                )  # split choices and corresponding indicators tuples again
                # multiple_choices = multiple_choices_indicators  # uncomment to debug
                correct_choice_idx = [
                    i
                    for i, (choice, indicator) in enumerate(
                        zip(multiple_choices, multiple_choices_indicators)
                    )
                    if indicator == correct_choice_indicator
                ]
                correct_choice = [multiple_choices[i] for i in correct_choice_idx]

            sample_dict[multiple_choice_enum] = (
                "".join(
                    [
                        f"{x} " if len(multiple_choices) == 2 else f"{x}, "
                        for x in symbols[:-1]
                    ]
                )
                + f"or {symbols[-1]}"
            )
            if self.multiple_choice_benchmarking_templates:
                if len(self.multiple_choice_rnd_symbols) > 1:
                    rnd_symbol = self.multiple_choice_rnd_symbols[
                        self.multiple_choice_benchmarking_format
                    ]
                elif len(self.multiple_choice_rnd_symbols) == 1:
                    rnd_symbol = self.multiple_choice_rnd_symbols[0]
                else:
                    raise NotImplementedError()
            else:
                rnd_symbol = random.sample(self.multiple_choice_rnd_symbols, k=1)[0]

            if rnd_symbol in ["()", "[]"]:
                rnd_symbol_prefix, rnd_symbol_suffix = rnd_symbol
            else:
                rnd_symbol_prefix = ""
                rnd_symbol_suffix = rnd_symbol

            multiple_choice_var_data = "\n".join(
                [
                    f"{rnd_symbol_prefix}{x}{rnd_symbol_suffix} {y}"
                    for x, y in zip(symbols, multiple_choices)
                ]
            )
            if multiple_choice_indicator == "":
                sample_dict[multiple_choice_var + "%"] = multiple_choice_var_data
            else:
                sample_dict[
                    multiple_choice_var + "%" + multiple_choice_indicator + "%"
                ] = multiple_choice_var_data

            # sample multiple_choice_result setup by randomly putting the result parts together
            # if self.multiple_choice_benchmarking_templates:
            # uncomment below to append correct_choice_idx with the symbols prefix and suffix
            # multiple_choice_result = f"{rnd_symbol_prefix}{symbols[correct_choice_idx]}{rnd_symbol_suffix}"
            # uncomment below to append correct_choice to the answer after the correct choice symbol
            # multiple_choice_result = f"{rnd_symbol_prefix}{symbols[correct_choice_idx]}{rnd_symbol_suffix} {correct_choice}"  # noqa: E501
            # else:
            # uncomment to include setup w/o symbols
            # if random.random() > 0.5:

            # uncomment to include setup w/ symbols
            # multiple_choice_result = symbols[correct_choice_idx]
            # if random.random() > 0.5:
            #    multiple_choice_result = (
            #        rnd_symbol_prefix + multiple_choice_result + rnd_symbol_suffix
            #    )

            # uncomment to include correct_choice
            # if random.random() > 0.5:
            #    if len(multiple_choice_result) > 0:
            #        multiple_choice_result += f" {correct_choice}"

            # uncomment to include setup w/o symbols
            # else:
            # multiple_choice_result = correct_choice

            if isinstance(correct_choice_idx, list):
                # correct_choice = "".join([str(x) for x in correct_choice])  # to get the full answer
                correct_choice = ", ".join([symbols[i] for i in correct_choice_idx])
                # correct_choice_idx = ", ".join([str(i) for i in correct_choice_idx])  # cast to str and join
            else:
                correct_choice = symbols[correct_choice_idx]
            multiple_choice_result = correct_choice

            sample_dict["%multiple_choice_result"] = multiple_choice_result

            # for benchmarking export
            sample_dict["%multiple_choice_symbols"] = symbols
            sample_dict["%multiple_choice_result_idx"] = correct_choice_idx

        # create sample dict
        for var in input_variables:
            if "#" in var:
                sample_dict[var] = self._get_target_from_row(sample, var)
            else:
                sample_dict[var] = get_target_from_string(self.meta, var)()

        return sample_dict

    def get_prompt_template_from_template_idx(self, template_idx: int = None) -> str:
        """Get prompt template from template index."""
        if template_idx is None:
            prompt_template = self.rnd_prompt_templates()
        else:
            prompt_template = self.prompt_templates[template_idx]
        return prompt_template

    def sample(self, sample: pd.Series, template_idx: int = None):
        """Sample text template by data from the sample row.
        The text template can be specified by the template index."""

        prompt_template = self.get_prompt_template_from_template_idx(template_idx)

        # if there are additional_targets we replace the SMILES randomly
        if len(self.additional_targets) > 0:
            # get additional targets that are not NaN for this sample
            non_nan_targets = (
                sample[["SMILES"] + self.additional_targets].dropna().keys().tolist()
            )
            new_target = random.sample(non_nan_targets, k=1)[0]
            if (
                new_target != "SMILES"
            ):  # if it is not SMILES we replace the corresponding parts in the prompt template
                # recreate prompt template object with replaced template to not change the original templates
                prompt_template = PromptTemplate(
                    prompt_template.template.replace("{SMILES", "{" + new_target)
                )

        sample_dict = self.get_sample_dict(sample, prompt_template.template)
        template = prompt_template.insert(sample_dict)

        if (
            self.benchmarking_templates
            and self.multiple_choice_benchmarking_templates
            and (any([k.startswith("%") for k in sample_dict]))
        ):
            # for multiple choice templates we need to keep track of the options and the correct answer
            # by appending them with special tokens to the end of the template.
            template += "<MC>" + "|".join(sample_dict["%multiple_choice_symbols"])
            if isinstance(sample_dict["%multiple_choice_result_idx"], list):
                template += "<MC>" + "|".join(
                    [str(x) for x in sample_dict["%multiple_choice_result_idx"]]
                )
            else:
                template += "<MC>" + str(sample_dict["%multiple_choice_result_idx"])

        return template

    def __getitem__(self, sample_idx: int, template_idx: int = None):
        """Get item from data with sample and template index.
        A random template will be ised if no template index is handed over."""
        sample = self.df.iloc[sample_idx]
        return self.sample(sample, template_idx)

    def apply_sampling(self, template_idx: int = None):
        """Applies the sampling to the entire data frame."""
        self.df["sample"] = self.df.apply(
            lambda sample: self.sample(sample, template_idx), axis=1
        )

    def export(self):
        """Exports the sampled data as separate jsonl files based on the split and benchmarking templates."""
        assert "sample" in self.df.columns, "Run apply_sampling before running export."
        print_data = {
            "split": [],
            "rows": [],
            "path": [],
        }
        for split in self.df.split.unique():
            # subselect for split
            df_out = self.df.copy()
            df_out = df_out[df_out["split"] == split]
            if self.benchmarking_templates:
                df_out[["input", "output"]] = df_out["sample"].str.split(
                    pat="<EOI>", n=1, expand=True
                )
                df_out.drop(
                    [
                        x
                        for x in df_out.columns.tolist()
                        if x not in ["input", "output"]
                    ],
                    axis=1,
                    inplace=True,
                )
                if self.multiple_choice_benchmarking_templates:
                    df_out[
                        ["output", "answer_choices", "correct_output_index"]
                    ] = df_out["output"].str.split(pat="<MC>", n=2, expand=True)
                    df_out["answer_choices"] = df_out["answer_choices"].apply(
                        lambda x: x.split("|")
                    )
                    df_out["correct_output_index"] = df_out[
                        "correct_output_index"
                    ].apply(lambda x: x.split("|"))
            else:
                df_out.drop(
                    [x for x in df_out.columns.tolist() if x != "sample"],
                    axis=1,
                    inplace=True,
                )
                df_out.columns = ["text"]

            # save
            if self.benchmarking_templates:
                # for lm eval harness we need to create yaml config files
                yaml.add_representer(str, str_presenter)
                yaml.representer.SafeRepresenter.add_representer(
                    str, str_presenter
                )  # to use with safe_dum

                if self.multiple_choice_benchmarking_templates:
                    output_path_dir = (
                        self.path_lm_eval_data_dir
                        + f"/{self.path_data_dir.split('/')[-1]}_benchmark_multiple_choice_format-{self.multiple_choice_benchmarking_format}/"  # noqa: E501
                    )
                    os.makedirs(output_path_dir, exist_ok=True)
                    output_path = output_path_dir + f"{split}.jsonl"

                    lm_eval_yaml_template_multiple_choice[
                        "task"
                    ] = self.path_data_dir.split("/")[-1]
                    lm_eval_yaml_template_multiple_choice[
                        "dataset_path"
                    ] = output_path_dir
                    lm_eval_yaml_template_multiple_choice[
                        "dataset_name"
                    ] = self.path_data_dir.split("/")[-1]

                    fn_lm_eval_yaml = output_path_dir + "/config.yaml"
                    with open(fn_lm_eval_yaml, "w") as f:
                        yaml.dump(
                            lm_eval_yaml_template_multiple_choice, f, sort_keys=False
                        )
                else:
                    output_path_dir = (
                        self.path_lm_eval_data_dir
                        + f"/{self.path_data_dir.split('/')[-1]}_benchmark/"
                    )
                    os.makedirs(output_path_dir, exist_ok=True)
                    output_path = output_path_dir + f"{split}.jsonl"

                    lm_eval_yaml_template_loglikelihood[
                        "task"
                    ] = self.path_data_dir.split("/")[-1]
                    lm_eval_yaml_template_loglikelihood[
                        "dataset_path"
                    ] = output_path_dir
                    lm_eval_yaml_template_loglikelihood[
                        "dataset_name"
                    ] = self.path_data_dir.split("/")[-1]

                    fn_lm_eval_yaml = output_path_dir + "/config.yaml"
                    with open(fn_lm_eval_yaml, "w") as f:
                        yaml.dump(
                            lm_eval_yaml_template_loglikelihood, f, sort_keys=False
                        )
            else:
                output_path_dir = (
                    self.path_lm_eval_data_dir
                    + f"/{self.path_data_dir.split('/')[-1]}/"
                )
                os.makedirs(output_path_dir, exist_ok=True)
                output_path = output_path_dir + f"{split}.jsonl"

            with open(output_path, "w") as f:
                f.write(df_out.to_json(orient="records", lines=True, force_ascii=False))

            # stats
            rows_split = len(df_out)
            print_data["split"].append(split)
            print_data["rows"].append(rows_split)
            print_data["path"].append(output_path)

        if len(self.df.split.unique()) > 1:
            print_data["split"].append("total")
            print_data["rows"].append(len(self.df))
            print_data["path"].append("")
        return pd.DataFrame(print_data)

    def apply_sampling_and_export(self, template_idx: int = None):
        """Applies the sampling and exports the data."""
        self.apply_sampling(template_idx)
        df_results = self.export()
        print(f"\n### results\n{df_results.to_string()}")


if __name__ == "__main__":
    path_base = __file__.replace("text_sampling/text_sampling.py", "")
    path_data_dir = sorted(glob.glob(path_base + "tabular/*")) + sorted(
        glob.glob(path_base + "kg/*[!.csv]")
    )
    path_lm_eval_data_dir = path_base + "text_sampling/export"

    for path in path_data_dir:
        print(f"\n###### {path}")
        path_meta = path + "/meta.yaml"
        path_data = path + "/data_clean.csv"
        if os.path.isfile(path_meta) and os.path.isfile(path_data):
            meta = load_yaml(path_meta)  # load yaml for downstream export logic

            # add standard text templates for tabular data
            additional_templates = None
            if (path.find("/tabular/") != -1) and not (
                any(
                    [
                        path.find(x) != -1
                        for x in exclude_from_standard_tabular_text_templates
                    ]
                )
            ):
                print("Add standard text templates for tabular data!")

                # if no SMILES identifier we continue
                if not (
                    any(
                        [
                            identifier["id"] == "SMILES"
                            for identifier in meta["identifiers"]
                        ]
                    )
                ):
                    print(
                        "No SMILES identifier in the meta.yaml. Please define custom text templates."
                    )
                    continue

                # if more than one target we continue
                # Note: More than one target needs custom text templates defined in the meta.yaml file.
                if len(meta["targets"]) > 1:
                    print(
                        "More than one target in the meta.yaml. Please define custom text templates."
                    )
                    continue

                # replace TARGET with target id
                additional_templates = [
                    template.replace("TARGET", meta["targets"][0]["id"])
                    for template in standard_tabular_text_templates
                ]
                if "templates" in meta:
                    meta["templates"] += additional_templates
                else:
                    meta["templates"] = additional_templates

            if "templates" in meta:
                multiple_choice_rnd_symbols = ["", ".", ".)", ")", ":", "()", "[]"]
                print(f"Running sampling for: {path}")
                TemplateSampler(
                    path,
                    path_lm_eval_data_dir,
                    multiple_choice_rnd_symbols=multiple_choice_rnd_symbols,
                    additional_templates=additional_templates,
                    benchmarking_templates=False,
                    multiple_choice_benchmarking_templates=False,
                ).apply_sampling_and_export()
                if any(["<EOI>" in t for t in meta["templates"]]):
                    TemplateSampler(
                        path,
                        path_lm_eval_data_dir,
                        multiple_choice_rnd_symbols=multiple_choice_rnd_symbols,
                        additional_templates=additional_templates,
                        benchmarking_templates=True,
                        multiple_choice_benchmarking_templates=False,
                    ).apply_sampling_and_export()
                    if any(["%multiple_choice_" in t for t in meta["templates"]]):
                        for i, s in enumerate(multiple_choice_rnd_symbols):
                            TemplateSampler(
                                path,
                                path_lm_eval_data_dir,
                                multiple_choice_rnd_symbols=[s],
                                additional_templates=additional_templates,
                                benchmarking_templates=True,
                                multiple_choice_benchmarking_templates=True,
                                multiple_choice_benchmarking_format=i,
                            ).apply_sampling_and_export()
