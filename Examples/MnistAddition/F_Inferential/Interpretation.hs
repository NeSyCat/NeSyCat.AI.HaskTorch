{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

-- | Inference layer (F) — INTERPRETATION for the MNIST example. MNIST's inference is
--   an INSTANCE of the (reused) 'InferenceSignature' wired to abstract library losses:
--   the knowledge penalty is @lossKnow = negLog@ on the [0,1] satisfaction object
--   @OmegaP@ — never a bespoke, MNIST-specific loss.
--
--   The training signal flows THROUGH the logic. The per-pair satisfaction is
--   @P(sum=n)@ (the @= n@ atom). The only quantifier — the @\forall@ over the data
--   pairs — is the logic's 'bigWedge', read in 'B_Logical.Interpretations.TensorProb'
--   as the PRODUCT t-norm aggregation (a geometric mean), the same reading MeasU gives
--   @\forall@. With @lossKnow = negLog@ this is exactly the categorical NLL
--   @J = -log(geomean_i P(sum=n_i)) = mean_i(-log P(sum=n_i))@, whose steep per-point
--   gradient avoids the collapsed optimum a soft @1 - SAT@ / p-mean-error falls into.
--
--   softmax (the push to probabilities) lives HERE, in the penalty — the categorical
--   analogue of binary's sigmoid — so GeomU stays logit-free. And @softmax@ over the
--   19 sum-logits is exactly @P(sum=n) = \sum_{d1+d2=n} softmax(lx)(d1) softmax(ly)(d2)@.
module MnistAddition.F_Inferential.Interpretation
  ( objective,
    trainConfig,
    batches,
  )
where

import A_Categorical.CategoricalInterpretation (GeomU)
import B_Logical.Interpretations.TensorProb (OmegaP (..))
import B_Logical.Signature.A2MonBLat (A2MonBLat (..))
import Data.Functor.Identity (Identity (..), runIdentity)
import F_Inferential.InferenceInterpretation () -- instance InferenceSignature OmegaP
import F_Inferential.InferenceSignature (InferenceSignature (..))
import MnistAddition.D_Grammatical.InterpretationTens (mnistSumLogits)
import MnistAddition.E_Data.Signature (MnistDataset (..))
import C_Domain.NeuralNets.DSL.Semantics (Weights)
import qualified Torch
import qualified Torch.Functional as F

-- | (epochs, learning rate). LTN's small-data single-digit setup: 20 epochs.
trainConfig :: (Int, Float)
trainConfig = (20, 0.001)

-- | Mini-batches of the training pairs (batch 64), re-SHUFFLED each epoch by a pure
--   per-epoch permutation @i \mapsto (a_e i + c_e) \bmod n@ (with @a_e@ coprime to
--   @n@), gathered via 'Torch.indexSelect''. Shuffling is good SGD hygiene here;
--   no RNG dependency -- the permutation is deterministic in @epoch@.
batches :: Int -> MnistDataset -> [(Torch.Tensor, Torch.Tensor, Torch.Tensor)]
batches epoch ds =
  let (xs, ys, hs) = trainBatch ds
      n = head (Torch.shape xs)
      mults = [997, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117]
      a = mults !! (epoch `mod` length mults) -- coprime to n (prime > 5), so the map is a bijection
      perm = [(a * i + 137 * epoch) `mod` n | i <- [0 .. n - 1]]
      gather t = Torch.indexSelect' 0 perm t
      (xs', ys', hs') = (gather xs, gather ys, gather hs)
      bs = 64
      slice t s = Torch.sliceDim 0 s (min (s + bs) n) 1 t
   in [(slice xs' s, slice ys' s, slice hs' s) | s <- [0, bs .. n - 1]]

-- | @J = -log SAT@ (the abstract 'lossKnow' = @negLog@ on the [0,1] satisfaction),
--   @SAT = bigWedge_{pairs} P(sum=n)@ = the product-t-norm @\forall@ (a geometric
--   mean). The epoch is unused (the product @\forall@ is parameter-free); the @1.0@ is
--   the inert @ParamsLogic@. Reaches the net only through @digit \@GeomU@ inside
--   'mnistSumLogits'.
objective :: Int -> (Torch.Tensor, Torch.Tensor, Torch.Tensor) -> Weights -> Torch.Tensor
objective _epoch (xB, yB, oneHotN) theta =
  let sumLogits = mnistSumLogits theta (xB, yB) -- [B,19] logits (the D term, logit-free)
      probs = F.softmax (F.Dim 1) sumLogits -- [B,19]  (softmax: only here, the penalty)
      s = Torch.sumDim (Torch.Dim 1) Torch.RemoveDim Torch.Float (oneHotN `Torch.mul` probs) -- [B] = P(sum=n)
      sat = runIdentity (bigWedge @Torch.Tensor @GeomU @OmegaP (1.0 :: Float) s (Identity . OmegaP)) -- forall = product
   in lossKnow sat -- = -log SAT
