package handlers

import (
	"context"
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

// workCheckBlock is how many hashes run between two context checks: about 0.2 ms
// of work, so a check (tens of nanoseconds) costs well under 0.1% of it.
const workCheckBlock = 4096

// chainedSHA256Ctx is chainedSHA256 that stops between blocks of workCheckBlock
// iterations once ctx is done, returning ctx.Err().
func chainedSHA256Ctx(ctx context.Context, iterations int) ([sha256.Size]byte, error) {
	var sum [sha256.Size]byte
	for i := 0; i < iterations; i++ {
		if i%workCheckBlock == 0 {
			if err := ctx.Err(); err != nil {
				return sum, err
			}
		}
		sum = sha256.Sum256(sum[:])
	}
	return sum, nil
}
