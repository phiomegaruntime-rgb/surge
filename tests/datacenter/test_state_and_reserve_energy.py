# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Regression checks for real solves, inter-period energy and reserve envelopes."""
from importlib import import_module

import pytest
import surge
from surge.market import REG_DOWN, REG_UP
from markets.datacenter import (
    AsProduct, BessSpec, DataCenterPolicy, DataCenterProblem, ItLoadSpec, SiteSpec,
)


def problem(*, initial=0., product=REG_UP, durations=(1.,1.), eta=.9):
    n=len(durations)
    return DataCenterProblem(
        period_durations_hours=list(durations), lmp_forecast_per_mwh=[100.]*n,
        site=SiteSpec(poi_limit_mw=100.,it_load=ItLoadSpec([50.]*n),
            bess=BessSpec(100.,100.,100.,initial_soc_mwh=initial,
                soc_min_fraction=0.,soc_max_fraction=1.,
                charge_efficiency=eta,discharge_efficiency=eta)),
        as_products=[AsProduct(product,[20.]*n)],
    )


def storage(period, rid='site_bess'):
    return next(r for r in period['resource_results'] if r['resource_id']==rid)


@pytest.mark.parametrize('product,initial',[(REG_UP,0.),(REG_UP,10.),(REG_DOWN,100.),(REG_DOWN,90.)])
@pytest.mark.parametrize('durations',[(1.,1.),(.25,.5,1.)])
def test_full_period_reserve_awards_have_storage_energy(product,initial,durations):
    p=problem(initial=initial,product=product,durations=durations)
    request=p.build_request(DataCenterPolicy(enforce_reserve_capacity=True))
    result=surge.solve_dispatch(p.build_network(),request,lp_solver='highs').to_dict()
    for dt,period in zip(durations,result['periods'],strict=True):
        r=storage(period);soc=r['detail']['soc_mwh'];award=r['reserve_awards'][product.id]
        if product.direction=='Up':
            assert award*dt/.9 <= soc+1e-6
        else:
            assert award*dt*.9 <= 100.-soc+1e-6


def test_guard_off_keeps_exploratory_behavior():
    p=problem(eta=1.)
    request=p.build_request(DataCenterPolicy(enforce_reserve_capacity=False))
    result=surge.solve_dispatch(p.build_network(),request,lp_solver='highs').to_dict()
    assert storage(result['periods'][0])['reserve_awards']['reg_up']==pytest.approx(100.)


@pytest.mark.parametrize('mode',['fixed','optimize'])
def test_datacenter_sequential_carries_zero_without_creating_energy(mode):
    p=problem(initial=100.,eta=1.,durations=(1.,1.,1.));p.as_products=[]
    policy=DataCenterPolicy(commitment_mode=mode,period_coupling='sequential')
    results=import_module('markets.datacenter.solve')._solve_sequential(p,policy)
    expected=100.
    for result in results:
        r=storage(result.to_dict()['periods'][0]);d=r['detail']
        expected+=d['charge_mw']-d['discharge_mw']
        assert d['soc_mwh']==pytest.approx(expected,abs=1e-6)
        assert expected>=-1e-6


def test_battery_sequential_carries_zero_without_creating_energy():
    from markets.battery import BatteryPolicy,BatteryProblem,SiteSpec as BatterySite
    p=BatteryProblem(period_durations_hours=[1.]*3,lmp_forecast_per_mwh=[100.]*3,
        site=BatterySite(poi_limit_mw=100.,bess_power_charge_mw=100.,bess_power_discharge_mw=100.,
            bess_energy_mwh=100.,bess_charge_efficiency=1.,bess_discharge_efficiency=1.,
            bess_soc_min_fraction=0.,bess_soc_max_fraction=1.,bess_initial_soc_mwh=100.,
            bess_degradation_cost_per_mwh=0.))
    results=import_module('markets.battery.solve')._solve_sequential(p,BatteryPolicy(period_coupling='sequential'))
    expected=100.
    for result in results:
        r=storage(result.to_dict()['periods'][0],p.BESS_RESOURCE_ID);d=r['detail']
        expected+=d['charge_mw']-d['discharge_mw']
        assert d['soc_mwh']==pytest.approx(expected,abs=1e-6)
        assert expected>=-1e-6
