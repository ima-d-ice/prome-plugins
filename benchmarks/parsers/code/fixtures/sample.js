import { logger } from './utils.js';

class TaskRunner {
    constructor(concurrency) {
        this.concurrency = concurrency;
    }

    runTask(taskId) {
        logger.info(`Running task ${taskId}`);
        return true;
    }
}

function processBatch(tasks) {
    return tasks.map(t => t.id);
}
export { TaskRunner, processBatch };