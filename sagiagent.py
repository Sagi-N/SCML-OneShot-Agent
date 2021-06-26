#!/usr/bin/env python
"""
**Submitted to ANAC 2021 SCML (OneShot track)**
*Authors* Sagi Nachum <sagi.nachum@gmail.com>


This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2021 SCML.

This module implements a factory manager for the SCM 2021 league of ANAC 2021
competition (one-shot track).
Game Description is available at:
http://www.yasserm.com/scml/scml2021oneshot.pdf

Your agent can sense and act in the world by calling methods in the AWI it has.
For all properties/methods available only to SCM agents check:
  http://www.yasserm.com/scml/scml2020docs/api/scml.oneshot.OneShotAWI.html

Documentation, tutorials and other goodies are available at:
  http://www.yasserm.com/scml/scml2020docs/

Competition website is: https://scml.cs.brown.edu

To test this template do the following:

0. Let the path to this file be /{path-to-this-file}/myagent.py

1. Install a venv (recommended)
>> python3 -m venv .venv

2. Activate the venv (required if you installed a venv)
On Linux/Mac:
    >> source .venv/bin/activate
On Windows:
    >> \.venv\Scripts\activate.bat

3. Update pip just in case (recommended)

>> pip install -U pip wheel

4. Install SCML

>> pip install scml

5. [Optional] Install last year's agents for STD/COLLUSION tracks only

>> pip install scml-agents

6. Run the script with no parameters (assuming you are )

>> python /{path-to-this-file}/myagent.py

You should see a short tournament running and results reported.

"""

# required for development
from scml.oneshot import OneShotAgent

# required for typing
from typing import Any, Dict, List, Optional

import numpy as np
from negmas import (
    AgentMechanismInterface,
    MechanismState,
    ResponseType,
)
from negmas.helpers import humanize_time
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE


# required for running tournaments and printing
import time
from tabulate import tabulate
from scml.scml2020.utils import anac2021_collusion, anac2021_oneshot, anac2021_std
from scml.oneshot.agents import (
    RandomOneShotAgent,
    SyncRandomOneShotAgent,
    GreedyOneShotAgent,
)


'''
The agent tries to maximize its revenues by constantly keeping track on the average sell price and average buy price it
has so far, and setting the acceptable buy prices and sell prices respectively – Always trying to sell at a price
higher than the average buy price and trying to buy at a price lower than the average sell price
'''
class SagiAgent(OneShotAgent):
    # =====================
    # Negotiation Callbacks
    # =====================
    '''
    This method implements the propose callback that is aimed at offering the negotiation partner thr acceptable deal.
    The method tries to offer sell prices at a price that higher than the current average but price, thus ensuring
    that the agent will have some revenue.
    In buy deals, the agent offers buy prices that are lower than the current average sell price.
    '''
    def propose(self, negotiator_id, state):
        """Called when the agent is asking to propose in one negotiation"""
        # self._outFile.write(f"\nProposed called!!! - Neg ID = {negotiator_id}\n")

        if not self._isExContractsInitialized: # Initialize ex contracts data for this step
            if self.awi.is_first_level and self.awi.current_exogenous_input_quantity > 0:
                self._MySecuredBuys = self.awi.current_exogenous_input_quantity
                self._MyAverageBuyPrice = self.awi.current_exogenous_input_price / self.awi.current_exogenous_input_quantity
                # self._outFile.write(f"Init ex contracts, _MySecuredBuys={self._MySecuredBuys}, _MyAverageBuyPrice={self._MyAverageBuyPrice}\n")
            elif self.awi.is_last_level and self.awi.current_exogenous_output_quantity > 0:
                self._MySecuredSells = self.awi.current_exogenous_output_quantity
                self._MyAverageSellPrice = self.awi.current_exogenous_output_price / self.awi.current_exogenous_output_quantity
                # self._outFile.write(f"Init ex contracts, _MySecuredSells={self._MySecuredSells}, _MyAverageSellPrice={self._MyAverageSellPrice}\n")


            self._isExContractsInitialized = True

        ami = self.get_ami(negotiator_id)
        if not ami:
            return None

        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]

        # Calculate buying/selling needs
        need_to_buy, need_to_sell = self._needs()
        isSelling = self._is_selling(ami)
        if isSelling: # This is a sell negotiation
            offer = [-1] * 3
            # Verify that the required quantity does not exceed the min/max allowed values
            quantityToOffer = max(min(need_to_sell, quantity_issue.max_value), quantity_issue.min_value)
            offer[QUANTITY] = quantityToOffer   # Set required quantity in response

            #  calculate the minimal acceptable selling price that will be put in offer
            MinAcceptableSellPrice, MinProfitablePrice = self.getMinAcceptableSellPrice(quantityToOffer, state, ami)

            #  Set acceptable price in response
            offer[UNIT_PRICE] = max(min(MinAcceptableSellPrice, unit_price_issue.max_value), unit_price_issue.min_value)

            offer[TIME] = self.awi.current_step
            # self._outFile.write(f"Proposing!!! - Neg ID = {negotiator_id}, quantity={offer[QUANTITY]}, price={offer[UNIT_PRICE]}\n")

            return tuple(offer)
        else:# This is a buying neotiation
            offer = [-1] * 3

            # Verify that the required quantity does not exceed the min/max allowed values
            quantityToOffer = max(min(need_to_buy, quantity_issue.max_value), quantity_issue.min_value)
            offer[QUANTITY] = quantityToOffer # Set required quantity in response

            #  calculate the maximal acceptable buying price that will be put in offer
            MaxAcceptableBuyPrice = self.getMaxAcceptableBuyPrice(state, ami)

            #  Set acceptable price in response
            offer[UNIT_PRICE] = max(min(MaxAcceptableBuyPrice, unit_price_issue.max_value), unit_price_issue.min_value)

            offer[TIME] = self.awi.current_step
            # self._outFile.write(f"Proposing!!! - Neg ID = {negotiator_id}, quantity={offer[QUANTITY]}, price={offer[UNIT_PRICE]}\n")

            return tuple(offer)

    '''
    This method implements the respond callback during a negotiation.
    The method follows the same logic that is done at the propose method. In case the partner wants to buy/sell in a 
    greater quantity than I'm willing to offer (Including the allowed buy/sell gap and maxAllowedInStock parameters) the
    offer is rejected, otherwise the agent checks that the offer price is acceptable (Such that guarantees having some
    profit) - If so, the agent accepts the offer
    '''
    def respond(self, negotiator_id, state, offer):

        """Called when the agent is asked to respond to an offer"""
        ami = self.get_ami(negotiator_id)
        if not ami:
            return None

        isSelling = self._is_selling(ami)

        #self._outFile.write(f"\nOffer received!!! - {role}. cur step={state.step}, max steps={ami.n_steps}, Neg ID={negotiator_id}\n")
        #self._outFile.write(f"Offer quantity={offer[QUANTITY]}, Offer price={offer[UNIT_PRICE]}, Offer time={offer[TIME]}\n")

        # find the quantity I still need and end negotiation if I need nothing more
        need_to_buy, need_to_sell = self._needs()
        if isSelling: # This is a sell negotiation
            if need_to_sell <= 0:
                #self._outFile.write(f"End negotiation - Need to sell={need_to_sell}\n")
                return ResponseType.END_NEGOTIATION # This is a selling offer but I don't need to sell

            if need_to_sell < offer[QUANTITY]:  # Partner wants more than I'm willing to sell
                #self._outFile.write(f"Rejecting offer - Need to sell={need_to_sell}, Offer quantity={offer[QUANTITY]}\n")
                return ResponseType.REJECT_OFFER # This is a selling offer but I don't have enough to sell
            else:   # The proposed quantity is fine, I can supply it
                quantityToOffer = offer[QUANTITY]
                isQuantityOK = True

            # Calculate minimum acceptable sell price
            mx = ami.issues[UNIT_PRICE].max_value  # Get max price
            MinAcceptableSellPrice, MinProfitablePrice = self.getMinAcceptableSellPrice(quantityToOffer, state, ami)
            if offer[UNIT_PRICE] > MinAcceptableSellPrice:
                isPriceOK = True    # Offer is more than the minimum acceptable sell price - Accept it
            elif offer[UNIT_PRICE] > 0.9 * mx:
                isPriceOK = True    # Offer is close to the maximum - Accept it
            else:   # Offered price is not OK - Reject offer
                #self._outFile.write(f"Rejecting offer - Need to sell={need_to_sell}, suggested unit price={offer[UNIT_PRICE]}, calculated min price={MinAcceptableSellPrice}\n")
                #self._outFile.write(f"th={self._th(state.step, ami.n_steps)}, mx={mx}, MinProfitablePrice={MinProfitablePrice}\n")
                #self._outFile.write(f"Avg buy price={self._MyAverageBuyPrice}, production cost={self._MyProductionCost}, Secured sells={self._MySecuredSells}, Avg sell price={self._MyAverageSellPrice}\n")
                return ResponseType.REJECT_OFFER

            if isQuantityOK and isPriceOK:  # Both suggested quantity and price are OK - Accept the offer
                #self._outFile.write(f"Accepting offer - Need to sell={need_to_sell}, Offer quantity={offer[QUANTITY]}, unit price={offer[UNIT_PRICE]}\n")
                return ResponseType.ACCEPT_OFFER
            else:
                #self._outFile.write(f"Rejecting offer - Need to sell={need_to_sell}, Offer quantity={offer[QUANTITY]}, unit price={offer[UNIT_PRICE]}\n")
                return ResponseType.REJECT_OFFER

        else: # This is a buy negotiation
            if need_to_buy <= 0:
                #self._outFile.write(f"End negotiation - Need to buy={need_to_buy}\n")
                return ResponseType.END_NEGOTIATION  # This is a buying offer but I don't need to buy

            if need_to_buy < offer[QUANTITY]:
                #self._outFile.write(f"Reject negotiation - Need to buy={need_to_buy}, Offer quantity={offer[QUANTITY]}\n")
                return ResponseType.REJECT_OFFER  # This is a buying offer but I don't need so much
            else:   # The proposed quantity is fine, I need it
                isQuantityOK = True

            # Calculate minimum acceptable sell price
            mn = ami.issues[UNIT_PRICE].min_value  # Get min price
            MaxAcceptableBuyPrice = self.getMaxAcceptableBuyPrice(state, ami)
            if offer[UNIT_PRICE] < MaxAcceptableBuyPrice:
                isPriceOK = True  # Offer is less than the maximum acceptable buy price - Accept it
            elif offer[UNIT_PRICE] < 1.1 * mn:
                isPriceOK = True  # Offer is close to the minimum - Accept it
            else:  # Price is not OK - Reject offer
                #self._outFile.write(
                #    f"Rejecting offer - Need to buy={need_to_buy}, suggested unit price={offer[UNIT_PRICE]}, calculated min price={MaxAcceptableBuyPrice}\n")
                #self._outFile.write(
                #    f"th={self._th(state.step, ami.n_steps)}, mn={mn}\n")
                #self._outFile.write(
                #    f"Avg buy price={self._MyAverageBuyPrice}, production cost={self._MyProductionCost}, Secured sells={self._MySecuredSells}, Avg sell price={self._MyAverageSellPrice}\n")
                return ResponseType.REJECT_OFFER

            if isQuantityOK and isPriceOK:  # Both suggested quantity and price are OK - Accept the offer
                #self._outFile.write(f"Accepting offer - Need to buy={need_to_buy}, Offer quantity={offer[QUANTITY]}, unit price={offer[UNIT_PRICE]}\n")
                return ResponseType.ACCEPT_OFFER
            else:
                #self._outFile.write(f"Rejecting offer - Need to buy={need_to_buy}, Offer quantity={offer[QUANTITY]}, unit price={offer[UNIT_PRICE]}\n")
                return ResponseType.REJECT_OFFER

    '''
    This method calculates the minimal sell price the agent is willing to accept in sell deals.
    The price is set taking intp consideration the average price of bought items, the production cost and a factor that
    determines the exact required value between that value and the max allowed value, tending to the lower price as the
    negotiatioon progresses
    '''
    def getMinAcceptableSellPrice(self, quantity, state, ami):
        totalSellPriceSoFar = self._MyAverageSellPrice * self._MySecuredSells   # Total sell price of items sold so far
        inStock = self._MySecuredBuys - self._MySecuredSells    # Amount of Items currently secured in stock (Buys-Sells)

        # Calculate the minimal profitable price that is based on the average buy price+production cost (That gives the
        # overall average cost of items currently in stock. Than ask for a sell price that is higher than that average
        # cost to guarantee having some profit
        MinProfitablePrice = ((self._MyAverageBuyPrice + self._MyProductionCost) * (self._MySecuredSells + quantity) - totalSellPriceSoFar) / quantity

        # Get price factor (Between 0-1), the factor determines the excat value of the required price between the minimal
        # profitable price and the max allowed price
        th = self._th(state.step, ami.n_steps)
        mx = ami.issues[UNIT_PRICE].max_value  # Get max price
        mn = ami.issues[UNIT_PRICE].min_value  # Get min price
        MinProfitablePrice = max(MinProfitablePrice, mn)

        # Determine acceptable price between min profitable price and max allowed price, tend to the min profitable
        # price as negotiation progresses
        MinAcceptableSellPrice = MinProfitablePrice + th * (mx - MinProfitablePrice)
        if (inStock > 0) and (ami.n_steps == state.step + 1):  # This is the last step in negotiation, get rid of stock
            #self._outFile.write("This is the last step, get rid of stock!!\n")
            MinAcceptableSellPrice = mn

        return MinAcceptableSellPrice, MinProfitablePrice

    '''
    This method calculates the maximal buy price the agent is willing to accept in buy deals.
    The price is set taking into consideration the average price of sold items and a factor that
    determines the exact required value between that value and the max allowed value, tending to the higher price as the
    negotiation progresses
    '''
    def getMaxAcceptableBuyPrice(self, state, ami):
        SellBuyGap = self._MySecuredSells - self._MySecuredBuys

        if SellBuyGap > 0:  # Agent is currently obliged to sell more than it actually has
            #I'm willing to buy at a price that I'm already obliged to sell minus my production cost minus the penalty I will get if I don't sell
            maxBuyPrice = self._MyAverageSellPrice - self._MyProductionCost
            th = self._th(state.step, ami.n_steps)
            mx = ami.issues[UNIT_PRICE].max_value  # Get max price
            maxBuyPrice = min(maxBuyPrice, mx) # Max buy price cannot be higher than the maximal price
            mn = ami.issues[UNIT_PRICE].min_value  # Get min price
            maxAcceptableBuyPrice = maxBuyPrice - th * (maxBuyPrice - mn)
            return maxAcceptableBuyPrice
        else:   # Agent is not obliged to sell items it does not have
            #I'm willing to buy at a max price as the average i bought so far
            mn = ami.issues[UNIT_PRICE].min_value  # Get min price
            maxBuyPrice = self._MyAverageBuyPrice
            if (maxBuyPrice < mn): # My average buy price is too low
                # The max price I'm willing to accept is the market trading price of my output product minus my production cost
                output_product = self.awi.my_output_product
                myOutputProductTradingPrice = self.awi.trading_prices[output_product]
                maxBuyPrice = myOutputProductTradingPrice - self._MyProductionCost

            maxBuyPrice = max(maxBuyPrice, mn)

            # Get price factor (Between 0-1), the factor determines the excat value of the required price between the maximal
            # profitable price and the max allowed price
            th = self._th(state.step, ami.n_steps)
            mn = ami.issues[UNIT_PRICE].min_value  # Get min price

            # Determine acceptable price between max profitable price and max allowed price, tend to the max profitable
            # price as negotiation progresses
            maxAcceptableBuyPrice = maxBuyPrice - th * (maxBuyPrice - mn)
            return maxAcceptableBuyPrice

    '''
    This method calculates a factor that determines how much the agent will be willing to compromise as the
    negotiation progresses. As much as negotiation progresses towards the end, the agent will be willing to accept
    compromise on the buy/sell price    
    '''
    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e

    # =====================
    # Time-Driven Callbacks
    # =====================
    '''
    This method is called when the agent is initialized
    '''
    def init(self):
        """Called once after the agent-world interface is initialized"""
        #self._outFile = open("out.txt", "a+")
        #self._outFile.write(f"Init called\n")

        self._isExContractsInitialized = False
        self._MySecuredBuys = 0
        self._MySecuredSells = 0
        self._MySecuredSells = 0
        self._MyStorageCost = self.awi.current_disposal_cost # penalizes buying too much/ selling too little
        self._MyDeliveryPenalty = self.awi.current_shortfall_penalty # penalizes buying too little / selling too much
        self._MyInputPenalty = self._MyStorageCost * self.awi.penalty_multiplier(True, 1.0)
        self._MyOutputPenalty = self._MyDeliveryPenalty * self.awi.penalty_multiplier(False, 1.0)
        self._MyProductionCost = self.awi.profile.cost  # Get item production cost
        self._MyAverageBuyPrice = 0.0   # Initialize average buy price
        self._MyAverageSellPrice = 0.0  # Initialize average sell price
        self._allowedBuySellGap = 5 # Allow a gap of up to 5 sold items more than buy items before stop selling
        self._maxAllowedInStock = 2 # Allow a max of 2 items in stock before stop buying
        self._e = 0.4   # A factor that is used during calculation of the th parameter

        #self._outFile.write(f"Total Products={self.awi.n_products}, Total Competitors={self.awi.n_competitors}\n")
        #self._outFile.write(f"My production cost={self._MyProductionCost}, number of lines={self.awi.n_lines}\n")
        #self._outFile.write(f"Input product={self.awi.my_input_product}, Output product={self.awi.my_output_product},Is First={self.awi.is_first_level}, Is Middle={self.awi.is_middle_level}, Is Last={self.awi.is_last_level}, Ex input={self.awi.current_exogenous_input_quantity}, ex output={self.awi.current_exogenous_output_quantity}, ex input price={self.awi.current_exogenous_input_price}, ex output price={self.awi.current_exogenous_output_price}\n")
        #self._outFile.write(f"My balance={myBalance}\n")
        #self._outFile.write(f"Penalties scale=={self.awi.penalties_scale}\n")
        #self._outFile.write(f"End init\n")

    '''
    This method is called at the beginning of every time step
    '''
    def step(self):
        """Called at every production step by the world"""
        self._isExContractsInitialized = False
        self._MySecuredBuys = 0
        self._MyAverageBuyPrice = 0.0
        self._MyAverageSellPrice = 0.0
        self._MySecuredSells = 0
        self._MyStorageCost = self.awi.current_disposal_cost # penalizes buying too much/ selling too little
        self._MyDeliveryPenalty = self.awi.current_shortfall_penalty # penalizes buying too little / selling too much
        self._MyInputPenalty = self._MyStorageCost * self.awi.penalty_multiplier(True, 1.0)
        self._MyOutputPenalty = self._MyDeliveryPenalty * self.awi.penalty_multiplier(False, 1.0)

        #self._outFile.write(f"\nStep {self.awi.current_step}: My storage cost={self._MyStorageCost}, My delivery penalty={self._MyDeliveryPenalty}\n")
        #self._outFile.write(f"Input product={input_product}, Output product={output_product},Is First={self.awi.is_first_level}, Is Middle={self.awi.is_middle_level}, Is Last={self.awi.is_last_level}, Ex input={self.awi.current_exogenous_input_quantity}, ex output={self.awi.current_exogenous_output_quantity}, ex input price={self.awi.current_exogenous_input_price}, ex output price={self.awi.current_exogenous_output_price}\n")
        #self._outFile.write(f"Trading price input={self.awi.trading_prices[input_product]}, Trading price output={self.awi.trading_prices[output_product]},Is First={self.awi.is_first_level}, Is Middle={self.awi.is_middle_level}, Is Last={self.awi.is_last_level}, Ex input={self.awi.current_exogenous_input_quantity}, ex output={self.awi.current_exogenous_output_quantity}\n")
        #self._outFile.write(f"Penalty multiplier input={self.awi.penalty_multiplier(True, 1.0)}, Penalty multiplier output={self.awi.penalty_multiplier(False, 1.0)}\n")
        #self._outFile.write(f"Max utility={self.ufun.max_utility}, Min utility={self.ufun.min_utility}\n")

        #self._outFile.write(f"Step {self.awi.current_step}, My balance={myBalance}\n\n")

    '''
    This method s called when negotiation succeeds.
    When negotiation succeeds the agent updates the amount of secured buys/sells it has and the average buy/sell
    price it has so far. Those parameters are used during negotiation when the agent determines that quantity it needs
    to buy/sell and price it is willing to accept.
    '''
    def on_negotiation_success(self, contract, mechanism):

        unit_price = contract.agreement["unit_price"]
        quantity = contract.agreement['quantity']

        if contract.annotation["product"] == self.awi.my_input_product: # This is a buying contract
            # Calculate new buy average
            curTotalPrice = self._MyAverageBuyPrice * self._MySecuredBuys
            curTotalPrice = curTotalPrice + (quantity * unit_price) # Add current contract to the total
            self._MySecuredBuys += quantity   # A buying contract
            self._MyAverageBuyPrice = curTotalPrice / self._MySecuredBuys   # Calculate new average
        else:   # This is a selling contract
            # Calculate new sell average
            curTotalPrice = self._MyAverageSellPrice * self._MySecuredSells
            curTotalPrice = curTotalPrice + (quantity * unit_price) # Add current contract to the total
            self._MySecuredSells += quantity  # A selling contract
            self._MyAverageSellPrice = curTotalPrice / self._MySecuredSells   # Calculate new average


        #self._outFile.write(f"\nNegotiation success!!! - {role}\n")
        #self._outFile.write(f"Input product={self.awi.my_input_product}, Output product={self.awi.my_output_product}\n")
        #self._outFile.write(
        #    f"Negotiation product={contract.annotation['product']}, quantity = {quantity}, unit price={unit_price}\n")
        #self._outFile.write(
        #    f"Ex input quantity={self.awi.current_exogenous_input_quantity}, Ex input unit price={self.awi.current_exogenous_input_price}\n")
        #self._outFile.write(
        #    f"Ex output quantity={self.awi.current_exogenous_output_quantity}, Ex output unit price={self.awi.current_exogenous_output_price}\n")

    '''
    This method calculates the amount of items the agent needs to buy/sell, taking into account exogenous contracs
    and the allowedBuySellGap (That determines how many items the agent is allowed to commit on selling even if it does
    not already secured their buy) and the maxAllowedInStock parameter (Which determines how many items tha agent is
    allowed to hold in stock even if it didn't already sold them)
    '''
    def _needs(self):
        """
        Returns both input and output needs
        """
        inStock = self._MySecuredBuys - self._MySecuredSells

        if self.awi.is_middle_level: # My agent is Mid level (It both buys and sells)
            summary = self.awi.exogenous_contract_summary

            # The amount I want to buy or sell is the minimum amount of total buys/sells for the current step
            return self._maxAllowedInStock - inStock, inStock + self._allowedBuySellGap

        if self.awi.is_first_level: # My agent is first level (Gets exogenous input and sells to next level)
            # No input needs. The amount I want to sell is the amount of exogenous input I got minus what I'm already obliged to sell
            return 0, self.awi.current_exogenous_input_quantity - self._MySecuredSells

        # No output needs. The amount I want to buy is the amount of exogenous output I have to provide minus the spllies I've already secured
        return self.awi.current_exogenous_output_quantity - self._MySecuredBuys, 0

    # ================================
    # Negotiation Control and Feedback
    # ================================

    def on_negotiation_failure(
        self,
        partners: List[str],
        annotation: Dict[str, Any],
        mechanism: AgentMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation the agent is a party of ends without
        agreement"""

    '''
    This method indicates to the agent whether current negotiation is on selling or buying deals
    '''
    def _is_selling(self, ami):
        if not ami:
            return None
        return ami.annotation["product"] == self.awi.my_output_product



def run(
    competition="oneshot",
    reveal_names=True,
    n_steps=10,
    n_configs=2,
):
    """
    **Not needed for submission.** You can use this function to test your agent.

    Args:
        competition: The competition type to run (possibilities are oneshot, std,
                     collusion).
        n_steps:     The number of simulation steps.
        n_configs:   Number of different world configurations to try.
                     Different world configurations will correspond to
                     different number of factories, profiles
                     , production graphs etc

    Returns:
        None

    Remarks:

        - This function will take several minutes to run.
        - To speed it up, use a smaller `n_step` value

    """
    if competition == "oneshot":
        competitors = [SagiAgent, RandomOneShotAgent, SyncRandomOneShotAgent, GreedyOneShotAgent]
    else:
        from scml.scml2020.agents import DecentralizingAgent, BuyCheapSellExpensiveAgent

        competitors = [
            SagiAgent,
            DecentralizingAgent,
            BuyCheapSellExpensiveAgent,
        ]

    start = time.perf_counter()
    if competition == "std":
        runner = anac2021_std
    elif competition == "collusion":
        runner = anac2021_collusion
    else:
        runner = anac2021_oneshot

    results = runner(
        competitors=competitors,
        verbose=True,
        n_steps=n_steps,
        n_configs=n_configs,
        #parallelism="serial",
    )
    # just make names shorter
    results.total_scores.agent_type = results.total_scores.agent_type.str.split(
        "."
    ).str[-1]
    # display results
    print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
    print(f"Finished in {humanize_time(time.perf_counter() - start)}")



if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "oneshot")
