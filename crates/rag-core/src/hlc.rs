//! Hybrid Logical Clock implementation for distributed sync.
//!
//! HLC combines physical wall-clock time with logical counters to provide
//! causally-ordered timestamps that are bounded by wall-clock time.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

/// Hybrid Logical Clock for causality tracking.
///
/// Format: 14 bytes
/// - Bytes 0-7: wall_time (big-endian u64, milliseconds since epoch)
/// - Bytes 8-11: logical (big-endian u32, logical counter)
/// - Bytes 12-13: node_id (big-endian u16)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HybridLogicalClock {
    /// Wall clock time in milliseconds since Unix epoch.
    pub wall_time: u64,

    /// Logical counter for events at the same wall time.
    pub logical: u32,

    /// Node identifier for tie-breaking.
    pub node_id: u16,
}

impl HybridLogicalClock {
    /// Create a new HLC with the current wall time.
    pub fn new(node_id: u16) -> Self {
        let wall_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            wall_time,
            logical: 0,
            node_id,
        }
    }

    /// Create an HLC from raw components.
    pub fn from_parts(wall_time: u64, logical: u32, node_id: u16) -> Self {
        Self {
            wall_time,
            logical,
            node_id,
        }
    }

    /// Create a zero/minimum HLC.
    pub fn zero() -> Self {
        Self {
            wall_time: 0,
            logical: 0,
            node_id: 0,
        }
    }

    /// Tick the clock for a local event.
    ///
    /// Returns a new HLC that is guaranteed to be greater than the current one.
    pub fn tick(&self) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        if now > self.wall_time {
            Self {
                wall_time: now,
                logical: 0,
                node_id: self.node_id,
            }
        } else {
            Self {
                wall_time: self.wall_time,
                logical: self.logical + 1,
                node_id: self.node_id,
            }
        }
    }

    /// Merge with a received HLC (on message receive).
    ///
    /// Returns a new HLC that is greater than both self and the received HLC.
    pub fn merge(&self, other: &Self) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let max_wall = now.max(self.wall_time).max(other.wall_time);

        let logical = if max_wall == self.wall_time && max_wall == other.wall_time {
            self.logical.max(other.logical) + 1
        } else if max_wall == self.wall_time {
            self.logical + 1
        } else if max_wall == other.wall_time {
            other.logical + 1
        } else {
            0
        };

        Self {
            wall_time: max_wall,
            logical,
            node_id: self.node_id,
        }
    }

    /// Convert to big-endian bytes for storage/comparison.
    ///
    /// The byte format allows lexicographic comparison in SQLite.
    pub fn to_bytes(&self) -> [u8; 14] {
        let mut buf = [0u8; 14];
        buf[0..8].copy_from_slice(&self.wall_time.to_be_bytes());
        buf[8..12].copy_from_slice(&self.logical.to_be_bytes());
        buf[12..14].copy_from_slice(&self.node_id.to_be_bytes());
        buf
    }

    /// Parse from big-endian bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 14 {
            return None;
        }

        let wall_time = u64::from_be_bytes(bytes[0..8].try_into().ok()?);
        let logical = u32::from_be_bytes(bytes[8..12].try_into().ok()?);
        let node_id = u16::from_be_bytes(bytes[12..14].try_into().ok()?);

        Some(Self {
            wall_time,
            logical,
            node_id,
        })
    }

    /// Convert to hex string for display.
    pub fn to_hex(&self) -> String {
        hex::encode(self.to_bytes())
    }

    /// Parse from hex string.
    pub fn from_hex(s: &str) -> Option<Self> {
        let bytes = hex::decode(s).ok()?;
        Self::from_bytes(&bytes)
    }
}

impl PartialEq for HybridLogicalClock {
    fn eq(&self, other: &Self) -> bool {
        self.to_bytes() == other.to_bytes()
    }
}

impl Eq for HybridLogicalClock {}

impl PartialOrd for HybridLogicalClock {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HybridLogicalClock {
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_bytes().cmp(&other.to_bytes())
    }
}

impl Default for HybridLogicalClock {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::fmt::Display for HybridLogicalClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlc_new() {
        let hlc = HybridLogicalClock::new(42);
        assert_eq!(hlc.node_id, 42);
        assert_eq!(hlc.logical, 0);
        assert!(hlc.wall_time > 0);
    }

    #[test]
    fn test_hlc_tick() {
        let hlc1 = HybridLogicalClock::new(1);
        let hlc2 = hlc1.tick();
        assert!(hlc2 > hlc1);
    }

    #[test]
    fn test_hlc_tick_rapid() {
        // Rapid ticks should increment logical counter
        let mut hlc = HybridLogicalClock::new(1);
        let initial_wall = hlc.wall_time;

        for _ in 0..10 {
            let next = hlc.tick();
            assert!(next > hlc);
            hlc = next;
        }

        // If wall time didn't advance, logical should have
        if hlc.wall_time == initial_wall {
            assert!(hlc.logical > 0);
        }
    }

    #[test]
    fn test_hlc_merge() {
        let local = HybridLogicalClock::from_parts(1000, 5, 1);
        let remote = HybridLogicalClock::from_parts(1000, 10, 2);

        let merged = local.merge(&remote);
        assert!(merged > local);
        assert!(merged > remote);
        assert_eq!(merged.node_id, 1); // Keeps local node_id
    }

    #[test]
    fn test_hlc_bytes_roundtrip() {
        let hlc = HybridLogicalClock::from_parts(1234567890, 42, 7);
        let bytes = hlc.to_bytes();
        let parsed = HybridLogicalClock::from_bytes(&bytes).unwrap();

        assert_eq!(hlc, parsed);
    }

    #[test]
    fn test_hlc_hex_roundtrip() {
        let hlc = HybridLogicalClock::from_parts(1234567890, 42, 7);
        let hex = hlc.to_hex();
        let parsed = HybridLogicalClock::from_hex(&hex).unwrap();

        assert_eq!(hlc, parsed);
    }

    #[test]
    fn test_hlc_ordering() {
        let a = HybridLogicalClock::from_parts(1000, 0, 1);
        let b = HybridLogicalClock::from_parts(1000, 1, 1);
        let c = HybridLogicalClock::from_parts(1001, 0, 1);

        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn test_hlc_bytes_lexicographic() {
        // Verify byte representation maintains ordering
        let a = HybridLogicalClock::from_parts(1000, 0, 1);
        let b = HybridLogicalClock::from_parts(1001, 0, 1);

        assert!(a.to_bytes() < b.to_bytes());
    }
}
