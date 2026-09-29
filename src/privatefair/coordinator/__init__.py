"""Track B: server-side coordination.

Privacy boundary (see interfaces.py and tests/test_boundaries.py): nothing in this
package may import privatefair.sim, privatefair.data, or reference TrueBins/sim_only.
It only ever sees privatized TelemetryReport payloads and its own server-side state.
"""
