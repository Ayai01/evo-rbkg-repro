"""Learn proposal value from within-root changes at exposed hardware and recovery choices."""

from bisect import bisect_left
from collections import Counter, defaultdict
from math import fsum, sqrt

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .cross_circuit_search import CrossCircuitSearch
from .source_frontier import CHUNK, _front_gain


class PairedDomainTransfer(CrossCircuitSearch):
    """Use a regularized within-root loss model to rank feasible physical moves.

    The source, exposed sites and recovery context are known before purchase.
    A root-specific intercept is removed by centering the purchased layouts of
    that root. Unseen native outcomes enter only through subsequent tell calls.
    """

    def __init__(self, circuit, *, root_metadata, projected_moves=True, adaptive_blocks=True, **settings):
        super().__init__(circuit, **settings)
        self.projected_moves = projected_moves
        self.adaptive_blocks = adaptive_blocks
        self.block_comparisons = 0
        self.profiles = {}
        self.profile_weights = Counter()
        self.cluster = {r: row.get('cluster', r) for r, row in root_metadata.items()}
        source_counts = Counter(row['source'] for row in root_metadata.values())
        schedules = {}
        for r, row in root_metadata.items():
            physical = row.get('physical', {})
            times = [e['cycle'] for e in physical.get('events', ()) if 'cycle' in e]
            if times:
                schedules[r] = max(times)
        source_times = defaultdict(list)
        for root, stamp in schedules.items():
            source_times[root_metadata[root]['source']].append(stamp)
        source_times = {s: sorted(values) for s, values in source_times.items()}
        for root, row in root_metadata.items():
            source = row['source']
            physical = row.get('physical', {})
            axes = physical.get('axes', circuit.source_axes[source])
            if physical.get('axis') is not None:
                axes = (physical['axis'],)
            phase = physical.get('phase')
            if root in schedules:
                times = source_times[source]
                phase = ('scheduled-third', 3*bisect_left(times, schedules[root])//len(times))
            profile = source, tuple(sorted(set(axes)-set(self.recovery))), phase
            self.profiles[root] = profile
            self.profile_weights[profile] += self.weights[source]/source_counts[source]
        self._fit_round = None
        self._fit = {}

    def _features(self, genome, profile):
        source, axes, phase = profile
        policy = tuple(genome[i] for i in self.recovery)
        features = {('policy',source,policy):1.}
        if phase is not None:
            features['phase-policy',source,phase,policy] = 1.
        scale = 1/sqrt(max(1,len(axes)))
        for i in axes:
            local = i,genome[i]
            features['protection',source,*local] = scale
            features['interaction',source,*local,policy] = scale
        return features

    def _coefficients(self):
        if self._fit_round==self._round:
            return self._fit
        self._fit_round = self._round
        by_root = self._labels_by_root()
        cluster_counts = Counter(self.cluster[r] for r, values in by_root.items() if len(values) >= 2)
        differences, response = [], []
        for root, cases in by_root.items():
            if len(cases)<2:
                continue
            vectors=[self._features(g,self.profiles[root]) for g in cases]
            center=Counter()
            for row in vectors:
                center.update(row)
            n=len(vectors)
            average=fsum(cases.values())/n
            # Replicated observations of one physical root have one total vote.
            scale=1/sqrt(n*cluster_counts[self.cluster[root]])
            for row,loss in zip(vectors,cases.values(),strict=True):
                values={key:(row.get(key,0.)-total/n)*scale for key,total in center.items()}
                values={key:v for key,v in values.items() if abs(v)>1e-12}
                if values:
                    differences.append(values)
                    response.append((loss-average)*scale)
        self._fit={}
        if not differences or not any(response):
            return self._fit
        index={key:i for i,key in enumerate(sorted({key for row in differences for key in row}, key=repr))}
        design=np.zeros((len(differences),len(index)))
        for i,row in enumerate(differences):
            for key,v in row.items():
                design[i,index[key]]=v
        # Fixed unit ridge, in root-equivalents and shared by every circuit.
        if design.shape[1]<=design.shape[0]:
            beta=np.linalg.solve(design.T@design+np.eye(len(index)),design.T@response)
        else:
            beta=design.T@np.linalg.solve(design@design.T+np.eye(len(design)),response)
        self._fit={key:float(beta[i]) for key,i in index.items()}
        return self._fit

    def _matched_guidance(self, parents, *, predict=None):
        if not self.projected_moves:
            return super()._matched_guidance(parents)
        beta=self._coefficients()
        if not beta:
            return super()._matched_guidance(parents)

        def costed_loss(genome):
            return fsum(weight*fsum(value*beta.get(k,0.) for k,value in self._features(genome,profile).items())
                        for profile,weight in self.profile_weights.items())

        predict = costed_loss if predict is None else predict
        possible=dict(self._catalog)
        if not self.circuit.candidates:
            # Propose true local substitutions and equal-area budget exchanges
            # from the physical axes involved in an observed root contrast.
            ranked=sorted(((abs(v),key) for key,v in beta.items() if key[0]=='protection'),reverse=True)
            axes=list(dict.fromkeys(key[2] for _,key in ranked))[:8]
            for parent in parents:
                moves=[(i,a) for i in (*axes,*self.recovery) for a in self.options[i] if a!=parent[i]]
                singles=[]
                for i,a in moves:
                    child=list(parent)
                    child[i]=a
                    value=self._geometry(tuple(child))
                    if value:
                        possible[value[0]]=value[1]
                        singles.append((i,a,value[1]-parents[parent]))
                for i,a,added in singles:
                    for j,b,released in singles:
                        if i==j or added<=0. or released>=0. or added+released>1e-8:
                            continue
                        child=list(parent)
                        child[i],child[j]=a,b
                        value=self._geometry(tuple(child))
                        if value:
                            possible[value[0]]=value[1]
        parent_scores={g:predict(g) for g in parents}
        scores={}
        span=self.circuit.reference-self.circuit.area_origin
        for genome,area in possible.items():
            if genome in self._areas:
                continue
            predicted=predict(genome)
            merit=max(((prior-predicted)/(1.+max(0.,area-parents[parent])/span)
                       for parent,prior in parent_scores.items()),default=0.)
            if merit>1e-10:
                scores[genome]=merit
        return scores

    def _offers(self, keys, limit, held):
        keys=list(keys)
        original=super()._offers(keys,limit,held)
        opening=any(any(r not in self._labels.get(g,{}) for r in self.blocks[b])
                    for g,b in self._opening if self._service.get(g) is not False)
        if not self.adaptive_blocks or opening or not original:
            return original
        by_block=defaultdict(list)
        for g,b in keys:
            by_block[b].append((g,b))
        if len(by_block)==1:
            return original
        layouts={g:self._geometry(g)[1] for g,_ in keys if self._service.get(g) is not False}
        active={g for g in layouts if self._labels.get(g)}
        active.update(p.genome for p in self._pending.values() if p.genome in layouts)
        rng=np.random.default_rng(self._seed(self._round,'fantasies'))
        shocks=ndtri(np.clip(qmc.Sobol(len(self.weights),rng=rng).random_base2(3),1e-10,1.-1e-10))
        shocks=np.vstack((shocks,-shocks))
        areas=np.array(list(layouts.values()))-self.circuit.area_origin
        reference=self.circuit.reference-self.circuit.area_origin
        best, choices={},{}
        for b, group in by_block.items():
            offers=super()._offers(group,limit,held)
            if not offers:
                continue
            location,mean,posts=self._posteriors(layouts,offers,sorted(active))
            admissible=[(location[g],b,offer[1]) for g,offer in offers.items()]
            incumbents={location[g] for g in active}
            merit=np.concatenate([_front_gain(mean,posts,self.weights,admissible[i:i+CHUNK],
                                      incumbents,areas,reference,shocks) for i in range(0,len(admissible),CHUNK)])
            for (g,offer),value in zip(offers.items(),merit,strict=True):
                score=float(value)/offer[3].median_cpu if self.mode.cost_aware else float(value)
                # Preserve declaration order on genuine ties.
                if g not in best or score>best[g]+1e-12:
                    best[g],choices[g]=score,offer
            self.block_comparisons+=len(admissible)
        return choices
