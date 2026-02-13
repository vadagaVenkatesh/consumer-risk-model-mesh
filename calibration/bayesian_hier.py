"""Bayesian Hierarchical Model

Implements hierarchical Bayesian inference for portfolio-level risk estimation.
Provides posterior distributions with uncertainty quantification.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats
from scipy.special import gammaln
import logging

logger = logging.getLogger(__name__)


class BayesianHierarchical:
    """
    Bayesian Hierarchical Model for Credit Risk
    
    Implements hierarchical Bayesian inference for multi-segment portfolios.
    Models segment-level default rates with portfolio-level hyperpriors.
    
    Mathematical Foundation:
    Hierarchical model structure:
    
    Hyperpriors (portfolio level):
        μ ~ Uniform(0, 1)
        κ ~ Gamma(a, b)
    
    Segment priors:
        θ_i ~ Beta(μκ, (1-μ)κ)  for i=1,...,K segments
    
    Likelihood:
        y_i ~ Binomial(n_i, θ_i)  where y_i = defaults in segment i
    
    Inference via Gibbs sampling or Variational Bayes.
    
    Attributes:
        method: Inference method ('gibbs' or 'vb')
        n_samples: Number of posterior samples
        segments: Segment identifiers
    """
    
    def __init__(
        self,
        method: str = 'gibbs',
        n_samples: int = 1000,
        burn_in: int = 200,
        random_state: Optional[int] = None
    ):
        """
        Initialize Bayesian hierarchical model.
        
        Args:
            method: Inference method ('gibbs' for MCMC, 'vb' for variational)
            n_samples: Number of posterior samples
            burn_in: Number of burn-in samples to discard
            random_state: Random seed for reproducibility
        """
        if method not in ['gibbs', 'vb']:
            raise ValueError(f"Method must be 'gibbs' or 'vb', got {method}")
        
        self.method = method
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.rng = np.random.default_rng(random_state)
        
        # Posterior samples
        self.theta_samples = None  # Segment-level parameters
        self.mu_samples = None     # Portfolio-level mean
        self.kappa_samples = None  # Portfolio-level concentration
        
        self.is_fitted = False
        
        logger.info(
            f"Initialized BayesianHierarchical with method={method}, "
            f"n_samples={n_samples}"
        )
    
    def fit(
        self,
        segment_defaults: np.ndarray,
        segment_exposures: np.ndarray,
        segment_ids: Optional[List[str]] = None
    ) -> 'BayesianHierarchical':
        """
        Fit hierarchical Bayesian model to portfolio data.
        
        Args:
            segment_defaults: Number of defaults per segment [K,]
            segment_exposures: Number of exposures per segment [K,]
            segment_ids: Optional segment identifiers
            
        Returns:
            Self for method chaining
        """
        segment_defaults = np.asarray(segment_defaults)
        segment_exposures = np.asarray(segment_exposures)
        
        if segment_defaults.shape != segment_exposures.shape:
            raise ValueError("Defaults and exposures must have same shape")
        if np.any(segment_defaults > segment_exposures):
            raise ValueError("Defaults cannot exceed exposures")
        
        K = len(segment_defaults)
        self.K = K
        self.segment_ids = segment_ids or [f"Seg_{i+1}" for i in range(K)]
        
        if self.method == 'gibbs':
            self._gibbs_sampling(segment_defaults, segment_exposures)
        else:
            self._variational_bayes(segment_defaults, segment_exposures)
        
        self.is_fitted = True
        logger.info(f"Fitted hierarchical model on {K} segments")
        
        return self
    
    def _gibbs_sampling(
        self,
        y: np.ndarray,
        n: np.ndarray
    ) -> None:
        """
        Gibbs sampling for posterior inference.
        
        Args:
            y: Defaults per segment
            n: Exposures per segment
        """
        K = len(y)
        total_samples = self.n_samples + self.burn_in
        
        # Initialize storage
        theta_chain = np.zeros((total_samples, K))
        mu_chain = np.zeros(total_samples)
        kappa_chain = np.zeros(total_samples)
        
        # Initialize parameters
        theta = self.rng.beta(1, 1, size=K)
        mu = 0.5
        kappa = 10.0
        
        # Hyperprior parameters
        a_kappa, b_kappa = 2.0, 0.1  # Gamma hyperprior for kappa
        
        # Gibbs sampling
        for t in range(total_samples):
            # Sample theta_i | y_i, n_i, mu, kappa
            alpha = mu * kappa + y
            beta = (1 - mu) * kappa + (n - y)
            theta = self.rng.beta(alpha, beta)
            
            # Sample mu | theta, kappa
            # Posterior is Beta(a_mu, b_mu)
            a_mu = kappa * np.sum(theta) + 1
            b_mu = kappa * (K - np.sum(theta)) + 1
            mu = self.rng.beta(a_mu, b_mu)
            
            # Sample kappa | theta, mu (Metropolis-Hastings step)
            kappa_proposal = np.exp(
                np.log(kappa) + self.rng.normal(0, 0.1)
            )
            
            log_accept = (
                self._log_kappa_posterior(kappa_proposal, theta, mu, a_kappa, b_kappa) -
                self._log_kappa_posterior(kappa, theta, mu, a_kappa, b_kappa)
            )
            
            if np.log(self.rng.uniform()) < log_accept:
                kappa = kappa_proposal
            
            # Store samples
            theta_chain[t] = theta
            mu_chain[t] = mu
            kappa_chain[t] = kappa
        
        # Discard burn-in
        self.theta_samples = theta_chain[self.burn_in:]
        self.mu_samples = mu_chain[self.burn_in:]
        self.kappa_samples = kappa_chain[self.burn_in:]
    
    def _log_kappa_posterior(
        self,
        kappa: float,
        theta: np.ndarray,
        mu: float,
        a: float,
        b: float
    ) -> float:
        """
        Log-posterior for kappa parameter.
        
        Args:
            kappa: Concentration parameter
            theta: Segment probabilities
            mu: Portfolio mean
            a, b: Gamma hyperprior parameters
            
        Returns:
            Log-posterior density
        """
        K = len(theta)
        
        # Log-likelihood from Beta priors
        alpha = mu * kappa
        beta = (1 - mu) * kappa
        
        log_lik = K * (
            gammaln(kappa) - gammaln(alpha) - gammaln(beta)
        ) + np.sum(
            (alpha - 1) * np.log(theta) + (beta - 1) * np.log(1 - theta)
        )
        
        # Log-prior: Gamma(a, b)
        log_prior = (a - 1) * np.log(kappa) - b * kappa
        
        return log_lik + log_prior
    
    def _variational_bayes(
        self,
        y: np.ndarray,
        n: np.ndarray
    ) -> None:
        """
        Variational Bayes approximation for faster inference.
        
        Uses mean-field approximation:
        q(θ, μ, κ) = ∏_i q(θ_i) q(μ) q(κ)
        
        Args:
            y: Defaults per segment
            n: Exposures per segment
        """
        K = len(y)
        
        # Variational parameters (point estimates)
        mu_var = 0.5
        kappa_var = 10.0
        
        # Coordinate ascent ELBO optimization
        for _ in range(100):
            # Update q(theta_i)
            alpha_var = mu_var * kappa_var + y
            beta_var = (1 - mu_var) * kappa_var + (n - y)
            
            # Update q(mu)
            theta_mean = alpha_var / (alpha_var + beta_var)
            a_mu = kappa_var * np.sum(theta_mean) + 1
            b_mu = kappa_var * (K - np.sum(theta_mean)) + 1
            mu_var = a_mu / (a_mu + b_mu)
        
        # Generate samples from variational distributions
        self.theta_samples = np.array([
            self.rng.beta(alpha_var, beta_var)
            for _ in range(self.n_samples)
        ])
        
        self.mu_samples = self.rng.beta(
            a_mu, b_mu, size=self.n_samples
        )
        
        self.kappa_samples = np.full(self.n_samples, kappa_var)
    
    def predict_segment(
        self,
        segment_idx: int,
        return_quantiles: bool = True
    ) -> Dict[str, float]:
        """
        Predict default rate for a segment with uncertainty.
        
        Args:
            segment_idx: Index of segment to predict
            return_quantiles: If True, return credible intervals
            
        Returns:
            Dict with posterior statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit model before prediction")
        if segment_idx >= self.K:
            raise ValueError(f"Invalid segment index {segment_idx}")
        
        samples = self.theta_samples[:, segment_idx]
        
        result = {
            'segment_id': self.segment_ids[segment_idx],
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples)
        }
        
        if return_quantiles:
            result.update({
                'q05': np.quantile(samples, 0.05),
                'q25': np.quantile(samples, 0.25),
                'q75': np.quantile(samples, 0.75),
                'q95': np.quantile(samples, 0.95)
            })
        
        return result
    
    def predict_new_segment(
        self,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict default rate for a new unseen segment.
        
        Uses posterior predictive distribution:
        p(θ_new | data) = ∫ p(θ_new | μ, κ) p(μ, κ | data) dμ dκ
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Samples from posterior predictive distribution
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit model before prediction")
        
        n = n_samples or self.n_samples
        
        # Sample from posterior predictive
        idx = self.rng.choice(len(self.mu_samples), size=n)
        mu = self.mu_samples[idx]
        kappa = self.kappa_samples[idx]
        
        alpha = mu * kappa
        beta = (1 - mu) * kappa
        
        theta_new = self.rng.beta(alpha, beta)
        
        return theta_new
    
    def portfolio_risk(
        self,
        exposures: np.ndarray,
        return_distribution: bool = False
    ) -> Dict:
        """
        Estimate portfolio-level risk with uncertainty.
        
        Args:
            exposures: Exposure amounts per segment [K,]
            return_distribution: If True, return full distribution
            
        Returns:
            Dict with portfolio risk statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit model before risk estimation")
        
        exposures = np.asarray(exposures)
        if len(exposures) != self.K:
            raise ValueError(f"Expected {self.K} exposures, got {len(exposures)}")
        
        # Portfolio loss distribution
        losses = self.theta_samples @ exposures
        
        result = {
            'expected_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'var_95': np.quantile(losses, 0.95),
            'var_99': np.quantile(losses, 0.99),
            'cvar_95': np.mean(losses[losses >= np.quantile(losses, 0.95)])
        }
        
        if return_distribution:
            result['loss_distribution'] = losses
        
        logger.info(
            f"Portfolio EL: {result['expected_loss']:.2f}, "
            f"VaR(95%): {result['var_95']:.2f}"
        )
        
        return result
