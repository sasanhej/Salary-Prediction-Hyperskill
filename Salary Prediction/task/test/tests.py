import re

from math import isclose

from hstest import CheckResult, StageTest, TestedProgram, WrongAnswer, dynamic_test


def is_float(num: str):
    try:
        float(num)
        return True
    except ValueError:
        return False


def check_outputs_number(values_number: int, user_output: str):
    outputs = user_output.split()

    if not all(is_float(output) for output in outputs):
        raise WrongAnswer(f"Answer '{user_output}' contains non-numeric values.")

    if len(outputs) != values_number:
        raise WrongAnswer(f"Answer contains {len(outputs)} values, but {values_number} values are expected.")


def check_digits(user_output: str, message: str):
    """Tests if a positive-negative number has no more than 5 digits after dot"""
    pattern = r"^[-]?[0-9]+.[0-9]{1,5}$"
    for value in user_output.split():
        m = re.match(pattern, value)
        if not m:
            raise WrongAnswer(message)


def check_num_values(values: list, user_values: list, message: str, rel_tol=1.0e-3):
    if not all(isclose(value, user_value, rel_tol=rel_tol) for value, user_value in zip(values, user_values)):
        raise WrongAnswer(message)


class Stage2Test(StageTest):
    @dynamic_test
    def test(self):
        pr = TestedProgram()
        user_output = pr.start().strip()

        if len(user_output) == 0:
            raise WrongAnswer("Seems like your program does not show any output.")

        # check output format
        check_outputs_number(1, user_output)

        # check digits
        check_digits(user_output, "Make sure that you provide numbers with no more than 5 digits after the dot.")

        # check values
        user_values = [float(value) for value in user_output.split()]

        mape_score = [0.9418168196379272]
        check_num_values(mape_score, user_values[:1],
                         "The submitted value (MAPE score) is wrong.",
                         rel_tol=1.0e-2)

        return CheckResult.correct()


if __name__ == '__main__':
    Stage2Test().run_tests()
