# Author: Jernej Frank


General comments:

- In addition to `poetry`, you also need to use something like `pyenv` to get the correct
python version 3.11 to make it run. Have you heard about `uv`? It is similar to `poetry`
but I find it much more straightforward to use and it is much faster. It's from the same
group that created `ruff` (alternative to `mypy` and `black` all in one).
- Added gitignore file as to not upload unnecessary things like caches.

1. test `is_a_counterexample`:

I saved the model weights of a run to make the test more deterministic. The function is
returning a boolean that is dependant on the model, but underlying structure of the
function should work independent of the model. So I use the same dummy input and ensure
that both boolean values are covered.

Given more time, I could make the add tests that ensure the model accuracy does drop
bellow a certain treshold using pytest-harvest to collect data on multiple re-trainings 
/ re-runs and then looking at the average accuracy of the `is_a_counterexample` output.

2. add `random_start` to `PGDAttack.pertube`:

I added an optional flag that allows to perturb the initial guess `x` by 
`n ~ U(-epsilon,epsilon)`. Since `torch.rand` samples from `[0,1)`, we just need to 
reparametrize the interval by `(max - min) * U(0,1) + min`.

I think the addition is straightforward enough that it does not need to be separate a
function with its own unit test. Given more time I would test the perturb function and 
also add some end-to-end tests that would capture the use case.

3. Calculate the robust accuracy:
 
I added the metric into `main`. If I understood the description correctly, a good model
would have high robustness and a more simple model would be susceptible to pgd. So in 
our case with the `SimpleNN` we expect the robustness accuracy to be very low.

In this example it is alright to keep it as a variable, but I imagine in a more serious
context we would be thinking about having something like a metric parent class and then
subclass it from there so that we can build a structured collection of different metrics
that can be chosen from to evaluate the models performance.



Some Afterthoughts:

- For the repo we could also do CI/CD to automatically run tests when merging to `main` 
(and test that all Python versions 3.8-3.13 are supported) and publish the changes.

- Use semantic versioning for more transparency with breaking changes to the APIs.

- Add a Changelog to keep track of changes.

- Now the model gets retrained every time. We can keep track of different parameter
inputs and safe the models for comparison using something like `mlflow`.

- If public, host docs for API reference and use-cases/concepts for an high-level 
overview how to use the package. Can also be automated in CI/CD. 

- Replace `print` statements with a `logger` for better control and experience.

- Containerising the application.