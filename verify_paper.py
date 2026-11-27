'''Recompute the main paper results from compact, frozen evaluation evidence.'''

import argparse
import gzip
import json
from math import comb, fsum, hypot, inf, isclose
from pathlib import Path
from statistics import mean

import numpy as np


CIRCUITS = ("fir", "rv32i", "conv-pool", "rocket")
AXES = ("FTE", "CPU_s")
PUBLIC = ("arh_clean", "qhvkg", "qlognehvi", "nsga2")
ABLATIONS = ("arh_clean", "no_evolution", "no_hierarchy", "no_rbkg")
METRICS = ("hv_auc", "terminal_hv", "cap_risk")
DATA = Path(__file__).with_name("paper-evidence.json.gz")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, context):
    check(isclose(actual, expected, rel_tol=0.0, abs_tol=5e-12), f"{context}: {actual} != {expected}")


def nondominated(points):
    front, lowest_risk = [], inf
    for area, risk in sorted(set(map(tuple, points))):
        if risk < lowest_risk:
            front.append((area, risk))
            lowest_risk = risk
    return front


def area_coordinate(area, geometry):
    return (area - geometry["area_origin"]) / (geometry["reference"] - geometry["area_origin"])


def hypervolume(points, geometry):
    x, risk, segments = 0., 1., []
    for area, next_risk in points:
        nxt = area_coordinate(area, geometry)
        if nxt >= 1.:
            break
        segments.append((nxt - x) * (1. - risk))
        x, risk = nxt, next_risk
    return fsum(segments + [(1. - x) * (1. - risk)])


def integral(curve, start=0.):
    endpoints = [x for x, _ in curve[1:]] + [1.]
    return fsum((right - max(left, start)) * value
                for (left, value), right in zip(curve, endpoints, strict=True)
                if right > max(left, start)) / (1. - start)


def exterior_distance(reference, points, geometry):
    target = [(area_coordinate(area, geometry), risk) for area, risk in reference]
    result = [(area_coordinate(area, geometry), risk) for area, risk in points]
    check(bool(result), "An IGD+ route has no terminal front")
    return mean(min(hypot(max(0., p_area - q_area), max(0., p_risk - q_risk))
                    for p_area, p_risk in result) for q_area, q_risk in target)


def replay_routes(circuits, seeds):
    all_endpoints, all_igd, attainment = {}, {}, {}
    for circuit in CIRCUITS:
        data = circuits[circuit]
        geometry, rows = data["geometry"], data["runs"]
        keys = [f"{seed}/{method}/{axis}" for seed in seeds for method in PUBLIC for axis in AXES]
        check(set(keys) == set(rows), f"{circuit}: incomplete route grid")
        pooled = nondominated(p for row in rows.values() for p in row["front"])
        check(len(pooled) == data["expected_reference_count"], f"{circuit}: pooled front size")
        for point, expected in zip(pooled, data["reference"], strict=True):
            close(point[0], expected[0], circuit)
            close(point[1], expected[1], circuit)
        for key in keys:
            row = rows[key]
            curves, opening = row["curve"], row["opening"]
            check(curves[0][0] == 0. and 0. <= opening < 1., key)
            check(all(0. <= x <= 1. and -5.e-12 <= v <= 1. for x, v in curves), key)
            check(all(b[0] >= a[0] for a, b in zip(curves, curves[1:])), key)
            check(data["budget"][key.rsplit("/",1)[1]] > 0, key)
            front = nondominated(row["front"])
            hv = hypervolume(front, geometry)
            close(hv, curves[-1][1], key)
            selected = row["selection"]
            check(selected is None or selected[0] <= geometry["cap"], f"{key}: out-of-cap selection")
            all_endpoints[circuit, key] = {"hv_auc": integral(curves), "terminal_hv": hv,
                                           "post_open_auc": integral(curves, opening),
                                           "cap_risk": 1. if selected is None else selected[1]}
            all_igd[circuit, key] = exterior_distance(pooled, front, geometry)
        attainment[circuit] = {}
        for method in PUBLIC:
            risks = [min((r for a,r in rows[f"{s}/{method}/FTE"]["front"] if a <= geometry["cap"]), default=inf)
                     for s in seeds]
            value = sorted(risks)[(len(seeds)-1)//2]
            attainment[circuit][method] = None if value == inf else value
    return all_endpoints, all_igd, attainment


def paired_effects(data, endpoints, seeds, family):
    summaries = []
    for circuit, axis, control, endpoint, expected, bounds in data["effects"][family]:
        if family == "public":
            candidate = [endpoints[circuit,f"{s}/arh_clean/{axis}"][endpoint] for s in seeds]
            comparator = [endpoints[circuit,f"{s}/{control}/{axis}"][endpoint] for s in seeds]
        else:
            panel = data["ablation_runs"][circuit][axis]
            candidate, comparator = panel["arh_clean"][endpoint], panel[control][endpoint]
        check(len(candidate) == len(comparator) == len(seeds), (family,circuit,axis,control,endpoint))
        sign = -1 if endpoint == "cap_risk" else 1
        value = mean(sign * (a - b) for a,b in zip(candidate,comparator,strict=True))
        close(value, expected, (family,circuit,axis,control,endpoint))
        summary = {"circuit":circuit, "axis":axis, "control":control, "endpoint":endpoint,
                   "pairs":len(seeds), "effect":value, "simultaneous_95":bounds}
        if endpoint == "hv_auc":
            summary["relative_percent"] = 100. * value / mean(comparator)
        summaries.append(summary)
    check(len(summaries) == len({(r["circuit"],r["axis"],r["control"],r["endpoint"]) for r in summaries}),
          f"{family}: duplicate comparison")
    return summaries


def igd_inference(spec, lookup, seeds):
    subjects = spec["bounds"]
    matrix = np.array([[lookup[c,f"{s}/{m}/{a}"] - lookup[c,f"{s}/arh_clean/{a}"] for s in seeds]
                       for c,a,m,_,_ in subjects])
    means = matrix.mean(axis=1)
    rng = np.random.default_rng(spec["seed"])
    replicates = np.empty((len(subjects), spec["draws"]))
    for circuit in CIRCUITS:
        rows = [i for i, row in enumerate(subjects) if row[0] == circuit]
        counts = rng.multinomial(len(seeds), np.full(len(seeds), 1. / len(seeds)), spec["draws"])
        replicates[rows, :] = (matrix[rows] @ counts.T) / len(seeds)
    scale = replicates.std(axis=1,ddof=1)
    check(np.all(scale > 0.), "Zero bootstrap scale in an IGD+ contrast")
    z = np.abs((replicates - means[:,None]) / scale[:,None])
    critical = float(np.quantile(z.max(axis=0), .95, method="higher"))
    close(critical, spec["critical"], "IGD+ joint critical value")
    results = []
    for i,(circuit,axis,control,expected,bounds) in enumerate(subjects):
        mean_value = float(means[i])
        radius = critical * float(scale[i])
        interval = [mean_value-radius, mean_value+radius]
        close(mean_value, expected, (circuit,axis,control))
        for actual,target in zip(interval,bounds,strict=True):
            close(actual, target, (circuit,axis,control))
        results.append({"circuit":circuit, "axis":axis, "control":control, "effect":mean_value,
                        "simultaneous_95":interval})
    return critical, results


def replay_sensitivity(rows, seeds):
    points = []
    for c,a,factor,level,e,set_value,default,paired,expected,interval,minimum,invariant in rows:
        check(len(paired)==len(seeds) and set_value!=default, (c,a,e,factor,level))
        effect = mean(paired)
        close(effect, expected, (c,a,e,factor,level))
        lower, upper = interval
        check(lower <= effect + 5e-12 and effect <= upper + 5e-12, (c,a,e,factor,level))
        if invariant:
            check(all(abs(val) < 5e-12 for val in (*paired,lower,upper)), (c,a,e,factor,level))
        points.append({"circuit":c,"axis":a,"factor":factor,"level":level,"endpoint":e,
                       "value":set_value,"default":default,"effect":effect,"simultaneous_95":interval,
                       "structural_invariant":invariant,"passes":lower>0 and effect>=minimum})
    eligible = [p for p in points if not p["structural_invariant"]]
    crosses = sum(p["simultaneous_95"][0]<=0<=p["simultaneous_95"][1] for p in eligible)
    improvements = sum(p["passes"] for p in eligible)
    check((len(points),len(eligible),crosses,improvements)==(480,468,463,0), "Sensitivity counts")
    return {"tested":len(points),"non_invariant":len(eligible),"zero_in_interval":crosses,
            "practical_improvements":improvements}, points


def exact_mcnemar(b,c):
    n=b+c
    return min(1.,2.*fsum(comb(n,k) for k in range(min(b,c)+1)) / 2**n)


def hardware_checks(data):
    fir_low,fir_high=data["hardware"]["fir"]
    rv_low,rv_high=data["hardware"]["rv32i"]
    close(fir_low["area_um2"],116057.396,"FIR selected layout")
    check(all(x["setup_slack_ns"] > 0 for x in (fir_low,fir_high,rv_low,rv_high)), "Mapped setup slack")
    area_saved=100.*(1.-fir_low["area_um2"]/fir_high["area_um2"])
    power_saved=[100.*(1.-a/b) for a,b in zip(fir_low["power_mW"],fir_high["power_mW"],strict=True)]
    check((round(area_saved,1),*[round(x,1) for x in power_saved])==(19.2,16.6,16.2), "FIR mapping headline")
    area_increase=100.*(rv_high["area_um2"]/rv_low["area_um2"]-1.)
    check(round(area_increase,1)==19.6,"RV32I area ordering")
    labels=data["mac_labels"]
    matches=sum(rtl==sdf and match for rtl,sdf,match in labels)
    check(len(labels)==matches==18,"MAC equivalence by archived case")
    cert=data["certificate"]
    lower=sum(cert["deltas"].values())
    expiry=cert["deadline"]-lower+1
    costs=cert["costs"]
    total=100.*(1.-costs["combined"]/costs["ordinary"])
    reused=100.*(1.-costs["shared"]/costs["ordinary"])
    added=100.*(1.-costs["combined"]/costs["shared"])
    check((lower,expiry,cert["error"],cert["timely"],cert["roots"])==(910415,310991,0,46,160),"Rocket certificate")
    check((round(total,1),round(reused,1),round(added,1))==(36.7,26.2,14.2),"Rocket execution cost")
    upper=100.*(1.-.05**(1./cert["timely"]))
    m=data["mechanism"]
    _,replay,resume,roots=m["rocket_replay"]
    initial,revised,b,c,n=m["rocket_recoding"]
    check(initial-revised==b-c and n==512 and roots==32,"Rocket matched counts")
    replay_p,recoding_p=exact_mcnemar(replay,resume),exact_mcnemar(b,c)
    check(replay_p==2.**-21 and recoding_p==2.**-19,"Rocket exact tests")
    g,images,doses,codes,modes=m["yolov8n"]
    sites_a,sites_b,mix_a,mix_b,mix=m["conv_pool_equal_area"]
    check(g*images*doses*codes*modes==640 and sites_a==sites_b==48 and mix_a==mix_b==144 and mix==192,
          "Controlled experiment denominators")
    return {"Table V count checks":{"Rocket replay vs resume p":replay_p,"Rocket recoding p":recoding_p,
                                   "YOLOv8n exact sign p":2.**-images,"Conv-Pool per-source exact sign p":2.**(1-sites_a)},
            "Table VI":{"FIR_area_saved_percent":area_saved,"FIR_power_saved_percent":power_saved,
                        "RV32I_area_increase_percent":area_increase,"MAC_matches":matches},
            "Rocket deadline and execution":{"lower_bound_cycles":lower,"first_certified_cycle":expiry,
                 "roots":cert["roots"],"stops":cert["stopped"],"erroneous_stops":cert["error"],
                 "timely_commits":cert["timely"],"exact_upper_percent":upper,
                 "reuse_saving_percent":reused,"increment_after_reuse_percent":added,"total_saving_percent":total}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json",type=Path,help="Write all computed endpoint and comparison rows to this file")
    args=p.parse_args()
    with gzip.open(DATA,"rt",encoding="utf-8") as source:data=json.load(source)
    seeds=data["seed_order"]
    endpoints,igd_lookup,attainment=replay_routes(data["curves"],seeds)
    public=paired_effects(data,endpoints,seeds,"public")
    ablation=paired_effects(data,endpoints,seeds,"ablation")
    critical,igd=igd_inference(data["igd"],igd_lookup,seeds)
    sensitivity,grid=replay_sensitivity(data["sensitivity"],seeds)
    hardware=hardware_checks(data)
    cap=[r for r in public if r["endpoint"]=="cap_risk"]
    cap_means=sum(r["effect"]>0 for r in cap)
    cap_bounds=sum(r["simultaneous_95"][0]>0 for r in cap)
    check((len(public),len(ablation),len(cap),cap_means,cap_bounds)==(78,72,24,24,17),"Paper Table II and IV counts")
    check(len(igd)==24 and sum(r["effect"]>0 for r in igd)==24 and
          sum(r["simultaneous_95"][0]>0 for r in igd)==12,"Paper Table III counts")
    check(round(attainment["conv-pool"]["arh_clean"]*100,1)==2.4 and
          round(attainment["rocket"]["arh_clean"]*100,1)==16.7,"Area-cap attainment in Fig. 3")
    result={"frozen_selection":data["selection_sha256"],"public_budget_traces":len(endpoints),
            "seeds_per_comparison":len(seeds),"public_effects_including_rocket_post_opening":len(public),
            "positive_mean_cap_risk_effects":cap_means,"positive_simultaneous_cap_risk_bounds":cap_bounds,
            "ablation_effects":len(ablation),"IGD_plus_draws":data["igd"]["draws"],
            "IGD_plus_comparisons":len(igd),"IGD_plus_joint_critical":critical,
            "FTE_area_cap_50_percent_attainment_risk":attainment,"sensitivity":sensitivity,**hardware}
    if args.json:
        full={**result,"public":public,"ablations":ablation,"IGD_plus":igd,
              "sensitivity_rows":grid,"endpoints":{f"{c}/{k}":v for (c,k),v in endpoints.items()}}
        args.json.write_text(json.dumps(full,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
