import { Middleware, RunAgentInput, AbstractAgent, BaseEvent } from '@ag-ui/client';
import { Observable } from 'rxjs';

interface EventThrottleConfig {
    /** Time-based throttle window in ms (e.g. 16 = ~60fps). */
    readonly intervalMs: number;
    /** Min new characters to accumulate before flushing. Default: 0. */
    readonly minChunkSize?: number;
}
declare class EventThrottleMiddleware extends Middleware {
    private readonly intervalMs;
    private readonly minChunkSize;
    private readonly isNoop;
    constructor(config: EventThrottleConfig);
    run(input: RunAgentInput, next: AbstractAgent): Observable<BaseEvent>;
}

export { type EventThrottleConfig, EventThrottleMiddleware };
