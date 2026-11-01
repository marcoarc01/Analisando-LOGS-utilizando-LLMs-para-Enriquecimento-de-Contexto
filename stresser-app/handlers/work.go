package handlers

import (
	"crypto/sha256"
	"sync/atomic"
)

// workSink keeps the compiler from discarding the per-request work.
var workSink atomic.Uint32

// chainedSHA256 hashes a 32-byte buffer `iterations` times. The cost is fixed
// in iterations, never in time, so CPU starvation makes it take longer.
func chainedSHA256(iterations int) [sha256.Size]byte {
	var sum [sha256.Size]byte
	for i := 0; i < iterations; i++ {
		sum = sha256.Sum256(sum[:])
	}
	return sum
}
