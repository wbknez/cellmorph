"""
Contains classes and functions pertaining to creating commonly used objects from
a configuration file.
"""
from dataclasses import astuple

from torch import Tensor
from torch.utils.data import RandomSampler, Sampler

from cellmorph.config import Configuration
from cellmorph.data import (
    GrowthStrategy,
    IndexingDataset,
    PersistentStrategy,
    RegenerativeStrategy,
    SamplePool,
    UpdateStrategy,
    empty_seed
)
from cellmorph.image import Dimension
from cellmorph.model import Model
from cellmorph.training import Trainer, load_target, prepare_target


class ConfigurationFactory:
    """
    Creates and configures common project-specific objects from a configuration
    file.
    """

    @classmethod
    def dataset(cls, config: Configuration,
                targets: Tensor | None = None) -> IndexingDataset:
        """
        Creates an :class:`IndexingDataset` based on values from a configuration
        file.

        Args:
            config: The configuration file to use.
            targets: The training targets; optional.

        Returns:
            A newly configured dataset.
        """
        if targets is None:
            targets = ConfigurationFactory.targets(config)

        return IndexingDataset(
            sample_count=config.data.sample_count, 
            state_channels=config.model.state_channels, 
            targets=targets
        )

    @classmethod
    def initial_state(cls, config: Configuration) -> Tensor:
        """
        Creates an initial state :class:`Tensor` based on model parameters from
        a configuration file for use with an :class:`UpdateStrategy`.

        Args:
            config: The configuration file to use.

        Returns:
            An initial state.
        """
        width, height = astuple(config.data.max_size)
        padding = config.data.padding

        size = Dimension(width + 2 * padding, height + 2 * padding)

        return empty_seed(config.model.state_channels, size)

    @classmethod
    def model(cls, config: Configuration) -> Model:
        """
        Creates a :class:`Model` based on values from a configuration file.

        Please note that this model has *neither* been loaded with any
        pretrained weights nor compiled via :meth:`torch.compile`.

        Args:
            config: The configuration to use.

        Returns:
            A newly configured model.
        """
        return Model(
            state_channels=config.model.state_channels,
            hidden_channels=config.model.hidden_channels,
            padding=config.model.padding,
            update_rate=config.model.update_rate,
            step_size=config.model.step_size,
            rotation=config.model.rotation,
            threshold=config.model.threshold,
            normalize_kernel=config.model.normalize_kernel,
            use_bias=config.model.use_bias
        )

    @classmethod
    def sample_pool(cls, config: Configuration,
                    ds: IndexingDataset | None = None,
                    sampler: Sampler | None = None) -> SamplePool:
        """
        Creates and configures a :class:`SamplePool` based on the values from a
        configuration file.

        Args:
            config: The configuration to use.
            ds: A collection of training samples and associated targets;
            optional.
            sampler: How samples should be selected per batch; optional.

        Returns:
            A newly configured sample pool.
        """
        if not ds:
            ds = ConfigurationFactory.dataset(config)

        if not sampler:
            sampler = RandomSampler(range(len(ds)))

        return SamplePool(ds, config.train.batch_size, sampler)

    @classmethod
    def strategy(cls, config: Configuration,
                 initial_state: Tensor | None = None) -> UpdateStrategy:
        """
        Creates and configures an :class:`UpdateStrategy` based on the
        "strategy" value from a configuration file.

        Args:
            config: The configuration to use.
            initial_state: An initial sample state; optional.

        Returns:
            A newly configured update strategy.
        """
        if initial_state is None:
            initial_state = ConfigurationFactory.initial_state(config)

        strategy_name = config.train.strategy

        reset_count = config.train.reset_count
        damage_count = config.train.damage_count

        match strategy_name.lower():
            case "growthstrategy" | "growth" | "grow":
                return GrowthStrategy(initial_state)
            case "persistentstrategy" | "persistent" | "persist":
                return PersistentStrategy(initial_state, reset_count)
            case "regenerativestrategy" | "regenerative" | "regenerate":
                return RegenerativeStrategy(initial_state, reset_count,
                                            damage_count)
            case _:
                raise ValueError(f"Unknown update strategy: {strategy_name}.")

    @classmethod
    def targets(cls, config: Configuration) -> Tensor:
        """
        Creates a collection of training targets from a configuration file.

        The returned :class:`Tensor` is repeated according to the number of
        samples for use with a dataset.

        Args:
            config: The configuration to use.

        Returns:
            A multi-dimensional target tensor.
        """
        target = config.data.target
        cache_dir = config.data.cache_dir

        with load_target(target, cache_dir=cache_dir) as img:
            target = prepare_target(img, config.data.max_size, config.data.padding,
                                    config.data.premultiply)

            return target.repeat((config.data.sample_count, 1, 1, 1))

    @classmethod
    def trainer(cls, config: Configuration, model: Model,
                strategy: UpdateStrategy | None = None) -> Trainer:
        """
        Creates a :class:`Trainer` based on values from a configuration file.

        Args:
            config: The configuration to use.
            strategy: The sample update strategy; optional.

        Returns:
            A newly configured trainer.
        """
        if not strategy:
            strategy = ConfigurationFactory.strategy(config)

        return Trainer(
            model=model,
            strategy=strategy,
            steps=config.train.steps,
            learning_rate=config.optim.learning_rate,
            milestones=config.optim.milestones,
            gamma=config.optim.gamma,
            gradient_cutoff=config.optim.gradient_cutoff
        )
