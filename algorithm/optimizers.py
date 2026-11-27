"""Categorical proposal operators extracted from the frozen optimizer."""

from itertools import accumulate


Genome = tuple[int, ...]


VARIATION_STRATEGIES = ("categorical_linkage", "categorical_independent")


PROTECTION_FAMILIES = frozenset(("memory_protection", "register_protection"))


NUMERICAL_FAMILIES = frozenset(("weight_precision", "activation_precision", "weight_grouping",
                                "accumulator_precision", "channel_weight_precision", "stage_activation_precision"))


class _Space:
    def __init__(self, options, linkage_levels, linkage_mode, rng,
                 variation_strategy="categorical_linkage", gene_families=(), coupled_groups=(), active_loci=None,
                 option_families=(), diversity_genome=None):
        self.options = tuple(tuple(values) for values in options)
        if not self.options or any(not values or any(type(v) is not int for v in values)
                                   or len(set(values)) != len(values) for values in self.options):
            raise ValueError("each nonempty domain must have distinct integer options")
        self.mutable = tuple(i for i, values in enumerate(self.options) if len(values) > 1)
        self.option_families = tuple(tuple(names) for names in option_families)
        if self.option_families and (len(self.option_families) != len(self.options)
                or any(len(names) != len(values) or any(type(name) is not str or not name for name in names)
                       for names, values in zip(self.option_families, self.options))):
            raise ValueError('option_families must name the mechanism of every allowed option')
        self.option_names = tuple(dict(zip(values, names)) for values, names in
                                  zip(self.options, self.option_families))
        self.active_loci = active_loci
        if diversity_genome is not None and not callable(diversity_genome):
            raise ValueError('diversity_genome must be a callable categorical projection')
        self.diversity_genome = diversity_genome
        self.rng = rng
        self.mode = linkage_mode
        if self.mode not in ("fine", "linked", "mixed"):
            raise ValueError("linkage_mode must be fine, linked or mixed")
        if variation_strategy not in VARIATION_STRATEGIES:
            raise ValueError(f"variation_strategy must be one of {VARIATION_STRATEGIES}")
        self.strategy = variation_strategy
        if gene_families and (len(gene_families) != len(self.options)
                              or any(not isinstance(value, str) or not value for value in gene_families)):
            raise ValueError("gene_families must contain one nonempty semantic name per domain")
        self.families = tuple(gene_families) if gene_families else ("categorical",) * len(self.options)
        self.typed_hardware = not PROTECTION_FAMILIES.isdisjoint(self.families)
        used = set()
        self.coupled_groups = []
        for raw in coupled_groups:
            group = tuple(raw)
            if (len(group) < 2 or len(set(group)) != len(group) or used.intersection(group)
                    or any(type(i) is not int or not 0 <= i < len(self.options) for i in group)):
                raise ValueError('Coupled groups require disjoint valid loci')
            used.update(group)
            self.coupled_groups.append(group)
        self.mutable_coupled_groups = [group for group in self.coupled_groups
                                       if any(len(self.options[i]) > 1 for i in group)]
        self.counts = dict.fromkeys(("crossovers", "crossover_independent", "crossover_linked",
                                    "mutations", "mutation_independent", "mutation_heterogeneous_block",
                                    "mutation_common_block", "nonzero_repairs", "mutation_memory",
                                    "mutation_register", "mutation_strategy", "mutation_subset_small",
                                    "mutation_subset_medium", "mutation_subset_large", "mutation_coupled", "crossover_coupled",
                                    "immigrant_numeric_reuse", "immigrant_numeric_small", "immigrant_numeric_medium",
                                    "immigrant_numeric_global"), 0)
        self.levels = []
        for level in linkage_levels:
            used, groups = set(), []
            for raw_group in level:
                group = tuple(raw_group)
                if (not group or any(type(i) is not int or not 0 <= i < len(self.options) for i in group)
                        or len(set(group)) != len(group) or used.intersection(group)):
                    raise ValueError("linkage groups within a level must be disjoint valid indices")
                used.update(group)
                # Equal integer codes do not imply equal semantics. Preserve
                # supplied family boundaries even in a coarse module group.
                by_family = {}
                for i in group:
                    namespace=(tuple(zip(self.options[i],self.option_families[i])) if self.option_families else ())
                    by_family.setdefault((self.families[i],namespace), []).append(i)
                for indices in by_family.values():
                    indices = tuple(indices)
                    common = tuple(v for v in self.options[indices[0]]
                                   if all(v in self.options[i] for i in indices))
                    if not common:
                        raise ValueError("linked domains in one family must share a compatible option")
                    groups.append((indices, common))
            groups.extend(((i,), self.options[i]) for i in range(len(self.options)) if i not in used)
            self.levels.append(tuple(groups))

    def validate(self, genome) -> Genome:
        genome = tuple(genome)
        if len(genome) != len(self.options) or any(type(v) is not int or v not in opts
                                                  for v, opts in zip(genome, self.options)):
            raise ValueError("genome has an unsupported option or incorrect domain count")
        return genome

    def random(self, progress, *, density=None, strategy_choices=None, parent=None) -> Genome:
        # Sample protection density as well as allele. Uniform independent
        # alleles alone overwhelmingly produce expensive dense large designs.
        density = self.rng.random() if density is None else density
        genome = []
        # Linked random module assignment caused a few all-module templates to
        # dominate initialization. Random immigrants always sample independent
        # decision units (the caller's engineering blocks), including density
        # and mechanism composition. Crossover still respects supplied linkage.
        if self.option_families:
            categories = {}
            for index, family in enumerate(self.families):
                categories.setdefault(family, {})
                categories[family].update(dict.fromkeys(self.option_families[index]))
            weights = {family: {name: self.rng.expovariate(1) for name in names}
                       for family, names in sorted(categories.items())}
        else:
            width = max(map(len, self.options))
            # Every axis with this family uses a prefix of the same weights.
            # Keep the draw for the unused base allele in the seeded stream.
            weights = {family: tuple(accumulate([self.rng.expovariate(1) for _ in range(width)][1:]))
                       for family in sorted(set(self.families))}
        for index, (values, family) in enumerate(zip(self.options, self.families)):
            if self.typed_hardware and family not in PROTECTION_FAMILIES:
                # Retry counts and reset protocols are independent categorical
                # choices; a low protection density must not bias them to zero.
                value = (strategy_choices[index] if strategy_choices is not None and index in strategy_choices
                         else self.rng.choice(values))
            else:
                if len(values) > 1 and self.rng.random() < density:
                    value = (self.choose_option(index, values[1:], weights[family]) if self.option_families
                             else self.rng.choices(values[1:], cum_weights=weights[family][:len(values)-1])[0])
                else:
                    value = values[0]
            genome.append(value)
        numeric = [i for i in self.mutable if self.families[i] in NUMERICAL_FAMILIES] if parent is not None else ()
        if numeric:
            # Protection and recovery immigrants need not rebuild an entirely
            # new numerical graph. Keep global, small and medium numeric moves.
            route = self.rng.choices(('reuse', 'small', 'medium', 'global'), (.425, .2125, .2125, .15))[0]
            self.counts['immigrant_numeric_'+route] += 1
            if route != 'global':
                for index in numeric:
                    genome[index] = parent[index]
                count = (0 if route == 'reuse' else self.rng.randint(1, min(4, len(numeric)))
                         if route == 'small' else self.rng.randint(min(5, len(numeric)), min(64, len(numeric))))
                for index in self.rng.sample(numeric, count):
                    self._change_gene(genome, index)
        return tuple(genome)

    def choose_option(self, index, values, weights=None):
        if not self.option_families:
            return self.rng.choice(values)
        groups = {}
        for value in values:
            groups.setdefault(self.option_names[index][value], []).append(value)
        names = tuple(groups)
        name = (self.rng.choices(names, weights=[weights[name] for name in names])[0]
                if weights is not None else self.rng.choice(names))
        return self.rng.choice(groups[name])

    def _change_gene(self, genome, index, changes=None):
        alternatives = [value for value in self.options[index] if value != genome[index]]
        genome[index] = self.choose_option(index, alternatives)
        if changes is not None:
            changes.add(index)

    def active_mutable(self, genome):
        if self.active_loci is None:
            return self.mutable
        indices = tuple(self.active_loci(tuple(genome)))
        if (len(indices) != len(set(indices)) or any(type(index) is not int or not 0 <= index < len(self.options) for index in indices)):
            raise ValueError('active_loci must return distinct declared genome indices')
        return tuple(index for index in indices if len(self.options[index]) > 1)
