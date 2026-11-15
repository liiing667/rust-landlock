// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::compat::try_compat_binary;
use crate::{
    private, AccessError, AddRuleError, AddRulesError, BitFlags, CompatError, CompatLevel,
    CompatResult, CompatState, Compatibility, HandleAccessError, HandleAccessesError, Ruleset,
    RulesetCreated, TailoredCompatLevel, TryCompat, ABI,
};
use enumflags2::BitFlag;
use std::marker::PhantomData;

#[cfg(test)]
use crate::{make_bitflags, AccessFs};
#[cfg(test)]
use strum::IntoEnumIterator;

pub trait Access: BitFlag + private::Sealed {
    /// Gets the access rights defined by a specific [`ABI`].
    fn from_all(abi: ABI) -> BitFlags<Self>;
}

// This HandledAccess trait is useful to document the API.
pub trait HandledAccess: Access {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Requested {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Effective {}

/// Requested or ABI-resolved ruleset attribute masks for one
/// [`Access`] category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RulesetMasks<A, S>
where
    A: Access,
{
    /// The handled access mask, or the scoped mask when `A` is
    /// [`Scope`](crate::Scope).
    pub(crate) access: BitFlags<A>,
    pub(crate) quiet: BitFlags<A>,
    _state: PhantomData<S>,
}

impl<A> Default for RulesetMasks<A, Requested>
where
    A: Access,
{
    fn default() -> Self {
        // Ruleset masks must start empty independently of BitFlags' customizable default.
        Self {
            access: BitFlags::EMPTY,
            quiet: BitFlags::EMPTY,
            _state: PhantomData,
        }
    }
}

impl<A> RulesetMasks<A, Requested>
where
    A: Access,
{
    const QUIET_SINCE: ABI = ABI::V10;

    pub(crate) fn supports_quiet(abi: ABI) -> bool {
        abi >= Self::QUIET_SINCE
    }

    pub(crate) fn request(
        &mut self,
        access: BitFlags<A>,
        compat: &mut Compatibility,
    ) -> Result<(), CompatError<A>> {
        // Keep this call for validation, compatibility-state transitions, and errors.
        // Its filtered mask is redundant: resolve() derives it from the ruleset mask and ABI.
        let requested = access;
        let _ = requested.try_compat(compat.abi(), compat.level, &mut compat.state)?;
        // Record the request only once validation succeeded: a rejected (e.g.
        // unknown or HardRequirement-incompatible) access must never mutate the
        // ruleset masks, which keeps error recovery and state transitions
        // deterministic.
        self.access |= access;
        Ok(())
    }

    pub(crate) fn request_quiet(
        &mut self,
        access: BitFlags<A>,
        compat: &mut Compatibility,
    ) -> Result<(), CompatError<A>> {
        // Keep both checks for their compatibility-state transitions and errors.
        let requested = access;
        let _ = requested.try_compat(compat.abi(), compat.level, &mut compat.state)?;
        let _ = compat.try_compat_binary(Self::supports_quiet(compat.abi()), || {
            CompatError::<A>::QuietNotSupported
        })?;
        // Same record-after-validation ordering as request(): an error must not
        // leave quiet bits (including unknown ones) recorded in the ruleset.
        self.quiet |= access;
        Ok(())
    }

    pub(crate) fn resolve(self, abi: ABI) -> RulesetMasks<A, Effective> {
        let effective_access = self.access & A::from_all(abi);
        RulesetMasks {
            access: effective_access,
            quiet: if Self::supports_quiet(abi) {
                self.quiet & effective_access
            } else {
                BitFlags::EMPTY
            },
            _state: PhantomData,
        }
    }

    pub(crate) fn incompatible_quiet(self) -> BitFlags<A> {
        self.quiet & negate_known(self.access)
    }
}

/// Bitwise negation restricted to the access rights known at build time.
///
/// Unlike `!flags`, this never yields undefined bits, so the result can be
/// stored back into a `BitFlags<A>` without an unsafe conversion.
pub(crate) fn negate_known<A>(flags: BitFlags<A>) -> BitFlags<A>
where
    A: Access,
{
    BitFlags::<A>::all() & !flags
}

/// Bitwise negation over the full underlying integer width, including
/// undefined bits.  The result must only be used as a mask in further bitwise
/// operations that feed back into a known-bit container (e.g. extracting
/// unknown bits); it is never stored into a ruleset mask.
pub(crate) fn negate_all_bits<A>(flags: BitFlags<A>) -> BitFlags<A>
where
    A: Access,
{
    // SAFETY: only consumed by intersections whose result is validated or
    // reported as an AccessError::Unknown; never recorded as a request.
    unsafe { BitFlags::<A>::from_bits_unchecked(!flags.bits()) }
}

/// ABI compatibility of an access mask against one kernel [`ABI`].
///
/// This is the single classification used both when recording a request
/// (state transitions and errors) and when deriving the effective masks sent
/// to the kernel, so both paths can never disagree about which bits are
/// supported by an ABI.
#[derive(Debug)]
pub(crate) enum AccessCompatibility<A>
where
    A: Access,
{
    /// Every requested bit is supported by the ABI.
    Full,
    /// Only a strict subset of the requested bits is supported.
    /// Downgrading can only narrow the mask, never widen it.
    Partial {
        supported: BitFlags<A>,
        incompatible: BitFlags<A>,
    },
    /// No requested bit is supported by the ABI.
    No,
}

/// Validates an access mask and classifies it against `abi`.
///
/// Empty masks and masks containing bits unknown at build time (only
/// reachable through the unsafe `BitFlags::from_bits_unchecked()`) are always
/// rejected, independently of the compatibility level and the ABI: unknown
/// bits must never be silently accepted nor merged into another mask.
pub(crate) fn classify_access<A>(
    access: BitFlags<A>,
    abi: ABI,
) -> Result<AccessCompatibility<A>, CompatError<A>>
where
    A: Access,
{
    if access.is_empty() {
        // Empty access-rights would result to a runtime error.
        return Err(AccessError::Empty.into());
    }
    if !BitFlags::<A>::all().contains(access) {
        // Unknown access-rights (at build time) would result to a runtime error.
        // This can only be reached by using the unsafe BitFlags::from_bits_unchecked().
        return Err(AccessError::Unknown {
            access,
            unknown: access & negate_all_bits(BitFlags::<A>::all()),
        }
        .into());
    }

    // The effective mask is always a subset of the request: intersection with
    // the rights defined by the ABI can only remove bits.
    let supported = access & A::from_all(abi);
    Ok(if supported.is_empty() {
        AccessCompatibility::No
    } else if supported == access {
        AccessCompatibility::Full
    } else {
        AccessCompatibility::Partial {
            supported,
            incompatible: access & negate_known(supported),
        }
    })
}

/// Applies the binary quiet-support part of a rule's compatibility handling.
///
/// `supported` tells whether the running ABI exposes the quiet-rule feature
/// at all; `effective` tells whether the ruleset still has an effective quiet
/// mask for the rule's category after ABI degradation.  Returns whether the
/// rule may actually be submitted with the quiet flag.  The
/// [`CompatState`] transitions are driven by [`try_compat_binary`], so every
/// rule category (filesystem and network) shares one predictable behavior.
pub(crate) fn quiet_rule_compat<A, L>(
    requested: bool,
    supported: bool,
    effective: bool,
    level: L,
    compat_state: &mut CompatState,
) -> Result<bool, CompatError<A>>
where
    A: Access,
    L: Into<CompatLevel>,
{
    if !requested {
        return Ok(false);
    }
    let mut quiet = try_compat_binary(supported, level, compat_state, || {
        CompatError::<A>::QuietNotSupported
    })?;
    // Quiet may be supported but have no effect after every quiet access in
    // this rule category has degraded away.  The ruleset compatibility state
    // already records that degradation, so the rule is simply not quiet.
    if quiet && !effective {
        quiet = false;
    }
    Ok(quiet)
}

#[test]
fn ruleset_masks_resolve() {
    let supports_quiet = RulesetMasks::<AccessFs, Requested>::supports_quiet;
    assert!(!supports_quiet(ABI::V9));
    assert!(supports_quiet(ABI::V10));

    let requested_masks = RulesetMasks::<_, Requested> {
        access: make_bitflags!(AccessFs::{Execute | Refer}),
        quiet: AccessFs::Execute.into(),
        _state: PhantomData,
    };
    for (abi, expected_access, expected_quiet) in [
        (ABI::Unsupported, BitFlags::EMPTY, BitFlags::EMPTY),
        (ABI::V1, AccessFs::Execute.into(), BitFlags::EMPTY),
        (
            ABI::V2,
            make_bitflags!(AccessFs::{Execute | Refer}),
            BitFlags::EMPTY,
        ),
        (
            ABI::V9,
            make_bitflags!(AccessFs::{Execute | Refer}),
            BitFlags::EMPTY,
        ),
        (
            ABI::V10,
            make_bitflags!(AccessFs::{Execute | Refer}),
            AccessFs::Execute.into(),
        ),
    ] {
        assert_eq!(
            requested_masks.resolve(abi),
            RulesetMasks::<_, Effective> {
                access: expected_access,
                quiet: expected_quiet,
                _state: PhantomData,
            }
        );
    }
}

pub trait PrivateHandledAccess: HandledAccess {
    fn ruleset_handle_access(
        ruleset: &mut Ruleset,
        access: BitFlags<Self>,
    ) -> Result<(), HandleAccessesError>
    where
        Self: Access;

    fn ruleset_quiet_access(
        ruleset: &mut Ruleset,
        access: BitFlags<Self>,
    ) -> Result<(), HandleAccessesError>
    where
        Self: Access;

    fn ruleset_has_effective_quiet_access(ruleset: &RulesetCreated, abi: ABI) -> bool;

    fn into_add_rules_error(error: AddRuleError<Self>) -> AddRulesError
    where
        Self: Access;

    fn into_handle_accesses_error(error: HandleAccessError<Self>) -> HandleAccessesError
    where
        Self: Access;
}

// Creates an illegal/overflowed BitFlags<T> with all its bits toggled, including undefined ones.
#[cfg(test)]
fn full_negation<T>(flags: BitFlags<T>) -> BitFlags<T>
where
    T: Access,
{
    negate_all_bits(flags)
}

#[test]
fn bit_flags_full_negation() {
    let scoped_negation = !BitFlags::<AccessFs>::all();
    assert_eq!(scoped_negation, BitFlags::<AccessFs>::empty());
    // !BitFlags::<AccessFs>::all() could be equal to full_negation(BitFlags::<AccessFs>::all()))
    // if all the 64-bits would be used, which is not currently the case.
    assert_ne!(scoped_negation, full_negation(BitFlags::<AccessFs>::all()));
}

impl<A> TailoredCompatLevel for BitFlags<A> where A: Access {}

impl<A> TryCompat<A> for BitFlags<A>
where
    A: Access,
{
    fn try_compat_inner(&mut self, abi: ABI) -> Result<CompatResult<A>, CompatError<A>> {
        let requested = *self;
        Ok(match classify_access(requested, abi)? {
            AccessCompatibility::Full => CompatResult::Full,
            AccessCompatibility::No => {
                CompatResult::No(AccessError::Incompatible { access: requested }.into())
            }
            AccessCompatibility::Partial {
                supported,
                incompatible,
            } => {
                // Mirror the historical in-place narrowing for callers that keep
                // the returned mask; resolve() remains the authoritative
                // derivation used for the kernel arguments.
                *self = supported;
                CompatResult::Partial(
                    AccessError::PartiallyCompatible {
                        access: requested,
                        incompatible,
                    }
                    .into(),
                )
            }
        })
    }
}

#[test]
fn compat_bit_flags() {
    use crate::ABI;

    let mut compat: Compatibility = ABI::V1.into();
    assert!(compat.state == CompatState::Init);

    let ro_access = make_bitflags!(AccessFs::{Execute | ReadFile | ReadDir});
    assert_eq!(
        ro_access,
        ro_access
            .try_compat(compat.abi(), compat.level, &mut compat.state)
            .unwrap()
            .unwrap()
    );
    assert!(compat.state == CompatState::Full);

    let empty_access = BitFlags::<AccessFs>::empty();
    assert!(matches!(
        empty_access
            .try_compat(compat.abi(), compat.level, &mut compat.state)
            .unwrap_err(),
        CompatError::Access(AccessError::Empty)
    ));

    let all_unknown_access = unsafe { BitFlags::<AccessFs>::from_bits_unchecked(1 << 63) };
    assert!(matches!(
        all_unknown_access.try_compat(compat.abi(), compat.level, &mut compat.state).unwrap_err(),
        CompatError::Access(AccessError::Unknown { access, unknown }) if access == all_unknown_access && unknown == all_unknown_access
    ));
    // An error makes the state final.
    assert!(compat.state == CompatState::Dummy);

    let some_unknown_access = unsafe { BitFlags::<AccessFs>::from_bits_unchecked((1 << 63) | 1) };
    assert!(matches!(
        some_unknown_access.try_compat(compat.abi(), compat.level, &mut compat.state).unwrap_err(),
        CompatError::Access(AccessError::Unknown { access, unknown }) if access == some_unknown_access && unknown == all_unknown_access
    ));
    assert!(compat.state == CompatState::Dummy);

    compat = ABI::Unsupported.into();

    // Tests that the ruleset is marked as unsupported.
    assert!(compat.state == CompatState::Init);

    // Access-rights are valid (but ignored) when they are not required for the current ABI.
    assert_eq!(
        None,
        ro_access
            .try_compat(compat.abi(), compat.level, &mut compat.state)
            .unwrap()
    );

    assert!(compat.state == CompatState::No);

    // Access-rights are not valid when they are required for the current ABI.
    compat.level = Some(CompatLevel::HardRequirement);
    assert!(matches!(
        ro_access.try_compat(compat.abi(), compat.level, &mut compat.state).unwrap_err(),
        CompatError::Access(AccessError::Incompatible { access }) if access == ro_access
    ));

    compat = ABI::V1.into();

    // Tests that the ruleset is marked as the unknown compatibility state.
    assert!(compat.state == CompatState::Init);

    // Access-rights are valid (but ignored) when they are not required for the current ABI.
    assert_eq!(
        ro_access,
        ro_access
            .try_compat(compat.abi(), compat.level, &mut compat.state)
            .unwrap()
            .unwrap()
    );

    // Tests that the ruleset is in an unsupported state, which is important to be able to still
    // enforce no_new_privs.
    assert!(compat.state == CompatState::Full);

    let v2_access = ro_access | AccessFs::Refer;

    // Access-rights are not valid when they are required for the current ABI.
    compat.level = Some(CompatLevel::HardRequirement);
    assert!(matches!(
        v2_access.try_compat(compat.abi(), compat.level, &mut compat.state).unwrap_err(),
        CompatError::Access(AccessError::PartiallyCompatible { access, incompatible })
            if access == v2_access && incompatible == AccessFs::Refer
    ));
}

#[test]
fn classify_access_abi_matrix() {
    // A fully supported mask keeps every bit.
    let v1 = AccessFs::from_all(ABI::V1);
    assert!(matches!(
        classify_access(v1, ABI::V1),
        Ok(AccessCompatibility::Full)
    ));

    // A mask mixing supported and unsupported bits is partial, and the
    // supported subset is strictly narrower than the request.
    let mixed = AccessFs::Execute | AccessFs::Refer;
    match classify_access(mixed, ABI::V1).unwrap() {
        AccessCompatibility::Partial {
            supported,
            incompatible,
        } => {
            assert_eq!(supported, BitFlags::<AccessFs>::from(AccessFs::Execute));
            assert_eq!(incompatible, BitFlags::<AccessFs>::from(AccessFs::Refer));
            assert!(mixed.contains(supported));
            assert!(supported != mixed);
        }
        other => panic!("expected Partial, got {other:?}"),
    }

    // Nothing supported on the mocked ABI.
    assert!(matches!(
        classify_access(AccessFs::Refer.into(), ABI::V1),
        Ok(AccessCompatibility::No)
    ));

    // Empty masks are always errors, regardless of the ABI.
    for abi in [ABI::Unsupported, ABI::V1, ABI::V10] {
        assert!(matches!(
            classify_access(BitFlags::<AccessFs>::EMPTY, abi),
            Err(CompatError::Access(AccessError::Empty))
        ));
    }
}

#[test]
fn classify_access_rejects_unknown_bits() {
    let unknown_only = unsafe { BitFlags::<AccessFs>::from_bits_unchecked(1 << 63) };
    assert!(matches!(
        classify_access(unknown_only, ABI::V10),
        Err(CompatError::Access(AccessError::Unknown { access, unknown }))
            if access == unknown_only && unknown == unknown_only
    ));

    // A known bit mixed with an unknown one is still rejected outright:
    // unknown bits must never be silently accepted or merged.
    let mixed = unsafe { BitFlags::<AccessFs>::from_bits_unchecked((1 << 63) | 1) };
    assert!(matches!(
        classify_access(mixed, ABI::V10),
        Err(CompatError::Access(AccessError::Unknown { access, unknown }))
            if access == mixed && unknown == unknown_only
    ));
}

#[test]
fn negate_helpers_keep_known_bits_scoped() {
    // negate_known only ever yields build-time known bits.
    let negated = negate_known(AccessFs::Execute.into());
    assert!(BitFlags::<AccessFs>::all().contains(negated));
    assert!(!negated.contains(AccessFs::Execute));

    // Full-width negation still surfaces unknown bits, which is exactly what
    // the Unknown error path needs to report.
    let all_negated = negate_all_bits(BitFlags::<AccessFs>::all());
    assert_eq!(all_negated & BitFlags::<AccessFs>::all(), BitFlags::EMPTY);
    assert_eq!(all_negated & unknown_only(), unknown_only());
}

#[cfg(test)]
fn unknown_only() -> BitFlags<AccessFs> {
    unsafe { BitFlags::<AccessFs>::from_bits_unchecked(1 << 63) }
}

#[test]
fn request_does_not_record_rejected_access() {
    // HardRequirement on an unsupported ABI: the request errors and must not
    // leave any trace in the handled mask.
    let mut masks = RulesetMasks::<AccessFs, Requested>::default();
    let mut compat: Compatibility = ABI::Unsupported.into();
    compat.level = Some(CompatLevel::HardRequirement);
    assert!(masks
        .request(AccessFs::Execute.into(), &mut compat)
        .is_err());
    assert!(masks.access.is_empty());
    assert_eq!(compat.state, CompatState::Dummy);

    // Unknown bits are rejected even under BestEffort and never recorded.
    let mut masks = RulesetMasks::<AccessFs, Requested>::default();
    let mut compat: Compatibility = ABI::V10.into();
    assert!(masks.request(unknown_only(), &mut compat).is_err());
    assert!(masks.access.is_empty());
    assert_eq!(compat.state, CompatState::Dummy);

    // A successful request is recorded as-is (resolve() narrows later).
    let mut masks = RulesetMasks::<AccessFs, Requested>::default();
    let mut compat: Compatibility = ABI::V1.into();
    masks
        .request(AccessFs::Execute | AccessFs::Refer, &mut compat)
        .unwrap();
    assert_eq!(masks.access, AccessFs::Execute | AccessFs::Refer);
}

#[test]
fn request_quiet_does_not_record_rejected_access() {
    // Quiet unsupported under HardRequirement: no quiet bit recorded.
    let mut masks = RulesetMasks::<AccessFs, Requested>::default();
    let mut compat: Compatibility = ABI::V9.into();
    compat.level = Some(CompatLevel::HardRequirement);
    assert!(matches!(
        masks.request_quiet(AccessFs::ReadFile.into(), &mut compat),
        Err(CompatError::QuietNotSupported)
    ));
    assert!(masks.quiet.is_empty());

    // Unknown quiet bits are rejected before the quiet-support check.
    let mut masks = RulesetMasks::<AccessFs, Requested>::default();
    let mut compat: Compatibility = ABI::V10.into();
    compat.level = Some(CompatLevel::HardRequirement);
    assert!(matches!(
        masks.request_quiet(unknown_only(), &mut compat),
        Err(CompatError::Access(AccessError::Unknown { .. }))
    ));
    assert!(masks.quiet.is_empty());

    // BestEffort on an ABI without quiet records the intent but resolve()
    // drops every quiet bit.
    let mut masks = RulesetMasks::<AccessFs, Requested>::default();
    let mut compat: Compatibility = ABI::V9.into();
    masks
        .request_quiet(AccessFs::ReadFile.into(), &mut compat)
        .unwrap();
    assert_eq!(masks.quiet, AccessFs::ReadFile);
    assert!(masks.resolve(ABI::V9).quiet.is_empty());
}

#[test]
fn resolve_only_narrows_and_quiet_stays_subset() {
    let masks = RulesetMasks::<AccessFs, Requested> {
        access: make_bitflags!(AccessFs::{Execute | Refer}),
        quiet: AccessFs::Execute.into(),
        _state: PhantomData,
    };

    for abi in ABI::iter() {
        let effective = masks.resolve(abi);
        // Effective masks can only be subsets of the requested ones.
        assert!(masks.access.contains(effective.access));
        assert!(masks.quiet.contains(effective.quiet));
        // Quiet can never include a bit absent from the effective handled mask.
        assert!(effective.access.contains(effective.quiet));
        // And quiet is only emitted when the ABI supports the quiet feature.
        if !RulesetMasks::<AccessFs, Requested>::supports_quiet(abi) {
            assert!(effective.quiet.is_empty());
        }
    }
}

#[test]
fn incompatible_quiet_only_reports_unhandled_known_bits() {
    let mut masks = RulesetMasks::<AccessFs, Requested> {
        access: AccessFs::ReadFile.into(),
        quiet: make_bitflags!(AccessFs::{ReadFile | WriteFile}),
        _state: PhantomData,
    };
    assert_eq!(
        masks.incompatible_quiet(),
        BitFlags::<AccessFs>::from(AccessFs::WriteFile)
    );

    // Fully handled quiet has nothing incompatible.
    masks.access = make_bitflags!(AccessFs::{ReadFile | WriteFile});
    assert!(masks.incompatible_quiet().is_empty());
}
