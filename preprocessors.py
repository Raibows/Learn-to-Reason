import os
import re
import random
from fewshot_demos import *
from tools import tools_json_load

def add_eos_token(sentences: list, eos_token: str) -> list:
    if eos_token is None:
        return sentences
    else:
        return [f"{s} {eos_token}" for s in sentences]

class DatasetProcessor:
    def __init__(self, eos_token, is_train_set, is_gpt, teacher_data_random=0.0, teacher_data=None, hint=False, hint_filter_path=None, fewshot=False, use_rationale=False):

        if is_train_set:
            # hint only works for inferring on the train set
            hint = False
            hint_filter_path = None

        self.is_train_set = is_train_set
        self.teacher_data_random = teacher_data_random
        self.teacher_data = teacher_data
        self.hint = hint
        self.hint_filter_path = hint_filter_path
        self.is_gpt = is_gpt
        self.fewshot = fewshot
        self.eos_token = eos_token
        self.use_rationale = use_rationale

        if hint:
            assert hint_filter_path is not None and os.path.exists(hint_filter_path) and teacher_data is None, f"{hint_filter_path}, {teacher_data}"
            self.filter_data = set()
            for item in tools_json_load(hint_filter_path)['results']:
                if not item['correct']:
                    self.filter_data.add(item['question'])
        else:
            self.filter_data = None

    def __call__(self, examples):
        raise NotImplementedError()

    def prepare_inputs(self, dataset_name: str, datas: list) -> dict:
        if len(datas) == 0:
            return {'inputs': [], 'prefix': []}

        assert 'question' in datas[0]
        assert 'answer' in datas[0]
        assert 'final_ans' in datas[0]

        res = [
            prepare_prompt(
                dataset=dataset_name,
                sample=item,
                fewshot=self.fewshot,
                add_hint=self.hint,
                use_rationale=self.use_rationale,
            )
            for item in datas
        ]

        prefix = [item[0] for item in res]

        if self.is_gpt and self.is_train_set:
            inputs = add_eos_token([item[1] for item in res], self.eos_token)
        else:
            inputs = prefix


        return {'inputs': inputs, 'prefix': prefix}




class MultiArithProcessor(DatasetProcessor):
    def __call__(self, examples):
        results = {
            'question': [],
            'answer': [],
            'final_ans': [],
        }
        for que, final_ans in zip(examples['question'], examples['final_ans']):
            que = que.strip()
            final_ans = str(final_ans).lower()
            prompt_answer = f"Therefore, the answer is {final_ans}."

            if isinstance(self.filter_data, set) and que not in self.filter_data:
                continue

            if self.teacher_data is None:
                results['question'].append(que)
                results['answer'].append(prompt_answer)
                results['final_ans'].append(final_ans)

            elif que in self.teacher_data:
                # control augment how much
                for aug_ans in self.teacher_data[que]:
                    if isinstance(aug_ans, str):
                        if random.random() < self.teacher_data_random:
                            results['question'].append(que)
                            results['answer'].append(aug_ans)
                            results['final_ans'].append(final_ans)

                    elif isinstance(aug_ans, dict):
                        for key in ['question', 'answer', 'final_ans']:
                            results[key].append(aug_ans.get(key))
                    else:
                        raise RuntimeError(f"{type(aug_ans)}\n{aug_ans}\nfailed")

        temp = self.prepare_inputs(
            'multiarith',
            [
                {'question': que, 'answer': ans, 'final_ans': fans}
                for que, ans, fans in zip(results['question'], results['answer'], results['final_ans'])
            ]
        )

        results['input'] = temp['inputs']
        results['prefix'] = temp['prefix']

        # gpt training input---target-------------------------infer input
        # question + answer   shift-question+answer          question

        # seq2seq training input---target---------------------infer input
        # question                 answer                    question

        if self.is_gpt:
            results['target'] = results['input']
        else:
            results['target'] = results['answer']

        return results



class SVAMPProcessor(DatasetProcessor):
    def __call__(self, examples):
        results = {
            'question': [],
            'answer': [],
            'final_ans': [],
        }

        for body, q, ans in zip(examples['Body'], examples['Question'], examples['Answer']):
            if not body.endswith('.'):
                body += '.'
            que = f"{body} {q}".strip()
            final_ans = str(int(ans)).lower()
            prompt_answer = f"Therefore, the answer is {final_ans}."

            if isinstance(self.filter_data, set) and que not in self.filter_data:
                continue

            if self.teacher_data is None:
                results['question'].append(que)
                results['answer'].append(prompt_answer)
                results['final_ans'].append(final_ans)

            elif que in self.teacher_data:
                # control augment how much
                for aug_ans in self.teacher_data[que]:
                    if isinstance(aug_ans, str):
                        if random.random() < self.teacher_data_random:
                            results['question'].append(que)
                            results['answer'].append(aug_ans)
                            results['final_ans'].append(final_ans)

                    elif isinstance(aug_ans, dict):
                        for key in ['question', 'answer', 'final_ans']:
                            results[key].append(aug_ans.get(key))
                    else:
                        raise RuntimeError(f"{type(aug_ans)}\n{aug_ans}\nfailed")

        temp = self.prepare_inputs(
            'svamp',
            [
                {'question': que, 'answer': ans, 'final_ans': fans}
                for que, ans, fans in zip(results['question'], results['answer'], results['final_ans'])
            ]
        )

        results['input'] = temp['inputs']
        results['prefix'] = temp['prefix']

        # gpt training input---target-------------------------infer input
        # question + answer   shift-question+answer          question

        # seq2seq training input---target---------------------infer input
        # question                 answer                    question

        if self.is_gpt:
            results['target'] = results['input']
        else:
            results['target'] = results['answer']

        return results


class CQAProcessor(DatasetProcessor):

    def __call__(self, examples):
        results = {'choices': [], 'question': [], 'final_ans': [], 'answer': [], 'input': [], 'target': []}

        for que, choices, fans in zip(examples['question'], examples['choices'], examples['answerKey']):

            if isinstance(self.filter_data, set) and que not in self.filter_data:
                continue

            fans = fans.lower()
            if self.teacher_data is None:
                results['choices'].append(choices)
                results['question'].append(que)
                results['answer'].append(fans)
                results['final_ans'].append(fans)
            elif que in self.teacher_data:
                for aug_ans in self.teacher_data[que]:
                    if isinstance(aug_ans, str):
                        if random.random() < self.teacher_data_random:
                            results['choices'].append(choices)
                            results['question'].append(que)
                            results['answer'].append(aug_ans)
                            results['final_ans'].append(fans)

                    elif isinstance(aug_ans, dict):
                        for key in ['question', 'answer', 'final_ans', 'choices']:
                            results[key].append(aug_ans.get(key))
                        if results['choices'][-1] is None: results['choices'][-1] = choices


        temp = self.prepare_inputs(
            'cqa', [{'choices': choices, 'question': que, 'answer': ans, 'final_ans': final_ans}
                    for que, choices, ans, final_ans in \
                        zip(results['question'], results['choices'], results['answer'], results['final_ans'])
                    ]
        )

        results['input'] = temp['inputs']
        results['prefix'] = temp['prefix']


        # gpt training input---target-------------------------infer input
        # question + answer   shift-question+answer          question

        # seq2seq training input---target---------------------infer input
        # question                 answer                    question

        if self.is_gpt:
            results['target'] = results['input']
        else:
            results['target'] = results['answer']

        return results



class GSM8KProcessor(DatasetProcessor):

    def __call__(self, examples):
        results = {
            'question': [],
            'answer': [],
            'final_ans': [],
            'original_answer': [],
        }

        for que, item in zip(examples['question'], examples['answer']):
            que = que.strip()
            t = item.strip().split("#### ")
            t[1] = t[1].strip().lower()
            t[0] = t[0].strip('\n').strip()
            answer = f"{t[0]} Therefore, the answer is {t[1]}."

            if isinstance(self.filter_data, set) and que not in self.filter_data:
                continue


            if self.teacher_data is None:
                results['question'].append(que)
                results['original_answer'].append(item)
                results['answer'].append(answer)
                results['final_ans'].append(t[1])

            elif que in self.teacher_data:
                # control augment how much
                for aug_ans in self.teacher_data[que]:
                    if isinstance(aug_ans, str):
                        if random.random() < self.teacher_data_random:
                            results['question'].append(que)
                            results['original_answer'].append(item)
                            results['answer'].append(aug_ans)
                            results['final_ans'].append(t[1])
                    elif isinstance(aug_ans, dict):
                        for key in ['question', 'original_answer', 'answer', 'final_ans']:
                            results[key].append(aug_ans.get(key))
                    else:
                        raise RuntimeError(f"{type(aug_ans)}\n{aug_ans}\nfailed")



        temp = self.prepare_inputs(
            'gsm8k',
            [
            {'question': que, 'answer': ans, 'final_ans': fans}
            for que, ans, fans in zip(results['question'], results['answer'], results['final_ans'])
            ]
        )

        results['input'] = temp['inputs']
        results['prefix'] = temp['prefix']

        # gpt training input---target-------------------------infer input
        # question + answer   shift-question+answer          question

        # seq2seq training input---target---------------------infer input
        # question                 answer                    question

        if self.is_gpt:
            results['target'] = results['input']
        else:
            results['target'] = results['answer']


        return results


class SQAProcessor(DatasetProcessor):
    def __call__(self, examples):
        results = {
            'question': [],
            'answer': [],
            'final_ans': [],
        }

        for que, ans in zip(examples['question'], examples['answer']):
            if ans:
                ans = "Yes"
            else:
                ans = "No"

            if isinstance(self.filter_data, set) and que not in self.filter_data:
                continue


            if self.teacher_data is None:
                results['question'].append(que)
                results['answer'].append(ans)
                results['final_ans'].append(ans)

            elif que in self.teacher_data:
                # control augment how much
                for aug_ans in self.teacher_data[que]:
                    if isinstance(aug_ans, str):
                        if random.random() < self.teacher_data_random:
                            results['question'].append(que)
                            results['answer'].append(aug_ans)
                            results['final_ans'].append(ans)

                    elif isinstance(aug_ans, dict):
                        for key in ['question', 'answer', 'final_ans']:
                            results[key].append(aug_ans.get(key))
                    else:
                        raise RuntimeError(f"{type(aug_ans)}\n{aug_ans}\nfailed")

        temp = self.prepare_inputs(
            'sqa',
            [
                {'question': que, 'answer': ans, 'final_ans': fans}
                for que, ans, fans in zip(results['question'], results['answer'], results['final_ans'])
            ]
        )

        results['input'] = temp['inputs']
        results['prefix'] = temp['prefix']

        # gpt training input---target-------------------------infer input
        # question + answer   shift-question+answer          question

        # seq2seq training input---target---------------------infer input
        # question                 answer                    question

        if self.is_gpt:
            results['target'] = results['input']
        else:
            results['target'] = results['answer']

        return results


def prepend_hint(text: str, final_ans):
    new_blocks = []
    cqa = 'Answer Choices' in text

    for block in text.split('\n\n'):
        hint_texts = []
        answer = None
        idx = None

        for line in block.split('\n'):
            if cqa:
                if line.startswith('Answer:'):
                    final_pred = re.findall(r'\(a\)|\(b\)|\(c\)|\(d\)|\(e\)', line)
                    if len(final_pred) > 0:
                        answer = f'{final_pred[0]}'
                    hint_texts.append('Hint: The final answer should be {final_ans}.')
                    idx = len(hint_texts) - 1
            else:
                if line.startswith('Reasoning:'):
                    hint_texts.append('Hint: The final answer should be {final_ans}.')
                    idx = len(hint_texts) - 1

                if line.startswith('Answer:'):
                    answer = line[len('Answer:'):].strip()

            hint_texts.append(line)

        assert idx is not None
        if answer is None:
            answer = final_ans if not cqa else f"({final_ans})"

        hint_texts[idx] = hint_texts[idx].format(final_ans=answer)
        new_blocks.append('\n'.join(hint_texts))

    return '\n\n'.join(new_blocks)


def prepare_prompt(dataset, sample: dict, add_hint=False, fewshot=False, use_rationale=False):
    if add_hint:
        assert 'final_ans' in sample
        assert fewshot
        assert use_rationale

    if dataset == 'cqa':
        choices = ''
        for label, text in zip(sample['choices']['label'], sample['choices']['text']):
            curr = '(' + label.lower() + ') ' + text

            # if add_hint and label.lower() == sample['final_ans'].lower():
            #     curr = f"{curr} (CORRECT)"
            # else:
            #     pass
            choices = choices + '\n' + curr

        if use_rationale and fewshot:
            # fewshot_demo = PROMPTS_CQA_HINT if add_hint else PROMPTS_CQA # poor performance
            fewshot_demo = PROMPTS_CQA
            prefix = fewshot_demo + '\n' + 'Question: ' + sample['question'] + f"\nAnswer Choices:{choices}" + '\nAnswer: '

            if add_hint: prefix = prepend_hint(prefix, sample['final_ans'])

            prefix_with_answer = prefix + sample['answer'].strip()

        elif use_rationale and not fewshot:
            prefix = BASELINE_PROMPTS['cqa'].format(question=sample['question'], choices=choices)
            prefix_with_answer = f"{prefix} {sample['final_ans']}"

        elif not (use_rationale or fewshot):
            prefix = BASELINE_PROMPTS['cqa'].format(question=sample['question'], choices=choices)
            prefix_with_answer = f"{prefix} ({sample['final_ans']})"

        elif not use_rationale and fewshot:
            prefix = BASELINE_PROMPT_FEWSHOT['cqa'] + '\n' + 'Question: ' + sample['question'] + f"\nAnswer Choices:{choices}" + '\nAnswer: '
            prefix_with_answer = f"{prefix}({sample['final_ans']})"

        else:
            raise NotImplementedError(f"rationale={use_rationale}, fewshot={fewshot}")


    elif dataset in {'gsm8k', 'svamp', 'multiarith', 'sqa'}:

        reasoning = sample['answer'].replace('\n', ' ').strip().replace('  ', '').strip()
        final_naswer = sample['final_ans']
        question = sample['question']

        if use_rationale and fewshot:
            PROMPT = {
                'gsm8k': PROMPTS_GSM8K,
                'svamp': PROMPTS_SVAMP,
                'multiarith': PROMPTS_MultiArith,
                'sqa': PROMPTS_SQA,
            }[dataset]

            prefix = f"{PROMPT}\nQuestion: {question}\nReasoning: "

            if add_hint: prefix = prepend_hint(prefix, sample['final_ans'])

            prefix_with_answer = f"{prefix}{reasoning}\nAnswer: {final_naswer}"

        elif use_rationale and not fewshot:
            prefix = f"Question: {question}\nReasoning: "
            prefix_with_answer = f"{prefix}{reasoning}\nAnswer: {final_naswer}"

        elif not (use_rationale or fewshot):
            prefix = BASELINE_PROMPTS[dataset].format(question=question)
            prefix_with_answer = f"{prefix} {final_naswer}"

        elif not use_rationale and fewshot:
            prefix = f"{BASELINE_PROMPT_FEWSHOT[dataset]}\nQuestion: {question}\nAnswer: "
            prefix_with_answer = f"{prefix}{final_naswer}"

        else:
            raise NotImplementedError(f"rationale={use_rationale}, fewshot={fewshot}")

    else:
        raise NotImplementedError()


    return prefix, prefix_with_answer
