import { Component, ViewChild, ElementRef, HostListener } from "@angular/core";
import { get } from "scriptjs";
import { NeuralNetwork } from "brain.js";
// import {ndarray} from 'ndarray';
const ndarray = require("ndarray");
const DQN = require("weblearn-dqn");
const { ReLU, Linear, MSE, SGD, Sequential } = require("weblearn");

import { interval, Subscription } from "rxjs";
@Component({
  selector: "my-app",
  templateUrl: "./app.component.html"
})
export class AppComponent {
  @ViewChild("myCanvas") canvasRef: ElementRef;

  private ctx: CanvasRenderingContext2D;
  private action: Move;
  private lastAction: Move;
  private snake: Coord[] = [];
  private foodSquare: Coord;
  private gridScale: number = 10;
  private HEIGHT: number = 100;
  private WIDTH: number = 100;
  private done: boolean = true;
  private score: number;
  private highScore: number = 0;
  private gameNo: number;
  private intervalSub: Subscription;
  private HUMAN_AGENT: boolean = true;
  private reward: number;
  private model: any;
  private agent: any;
  private doneTraining: boolean = false;
  private currentDistanceToFood: number = 99999999;
  private movesSinceLastEating: 0;
  private lastTransitions: LastTransitionsIndexes = {
    lowPriority: [],
    highPriority: []
  };
  @HostListener("window:keydown", ["$event"])
  handleKeyboardEvent(event: KeyboardEvent) {
    let previousSnakeHead = this.snake[this.snake.length - 1];

    switch (event.keyCode) {
      case 37: {
        if (this.action != Move.RIGHT) this.action = Move.LEFT;
        break;
      }
      case 39: {
        if (this.action != Move.LEFT) this.action = Move.RIGHT;
        break;
      }
      case 38: {
        if (this.action != Move.DOWN) this.action = Move.UP;
        break;
      }
      case 40:
        {
          if (this.action != Move.UP) this.action = Move.DOWN;
          break;
        }
        this.makeMove();
        setTimeout(() => {
          this.redraw();
        });
    }
  }

  ngOnInit() {
    this.ctx = this.canvasRef.nativeElement.getContext("2d");
    this.reset();
    setTimeout(() => {
      this.redraw();
    });
    this.model = Sequential({
      optimizer: SGD(0.01),
      loss: MSE()
    });
    const STATE_SIZE = 32;
    const NUM_ACTIONS = 4;

    this.model
      .add(Linear(STATE_SIZE, 32, false))
      .add(ReLU())
      .add(Linear(32, 32))
      .add(ReLU())
      .add(Linear(32, 32))
      .add(ReLU())
      .add(Linear(32, NUM_ACTIONS, false));
 
    this.agent = DQN({
      model: this.model,
      numActions: NUM_ACTIONS,
      finalEpsilon: 0.1,
      epsilonDecaySteps: 10000,
      memorySize: 10000,
      gamma: 0.9
    });
    this.agent.transitionsLowPriority = [];
    this.agent.theta = 0.8;

    this.agent.step = (currentState, currentReward, done) => {
      let currentAction;
      if (Math.random() < this.agent.epsilon) {
        currentAction = this.getAllPossibleMoves()[
          Math.floor(Math.random() * (this.agent.numActions - 1))
        ];
      } else {
        const modelOutputs = this.model.forward(currentState);
        currentAction = this.indexOfMax(modelOutputs.data);
      }

      let trainingGap: number = this.getTrainingGap();
      let withinTrainingGap: boolean = this.movesSinceLastEating > 0 && this.movesSinceLastEating < trainingGap;

      if (this.agent.previousState && typeof currentReward === "number", !withinTrainingGap) {
        const transition = [
          this.agent.previousState,
          this.agent.previousAction,
          currentReward,
          currentState,
          done
        ];
        if (currentReward >= 0.5) {
          this.lastTransitions.highPriority.push(
            this.agent.transitions.length % (this.agent.memorySize / 2)
          );
          this.agent.transitions[
            this.agent.transitions.length % (this.agent.memorySize / 2)
          ] = transition;
        } else {
          this.lastTransitions.lowPriority.push(
            this.agent.transitionsLowPriority.length %
              (this.agent.memorySize / 2)
          );
          this.agent.transitionsLowPriority[
            this.agent.transitionsLowPriority.length %
              (this.agent.memorySize / 2)
          ] = transition;
        }
        this.agent.transitionCount++;
      }
      // console.log('low',this.agent.transitionsLowPriority.length);
      // console.log('high',this.agent.transitions.length);
      if (this.reward == 1 || this.reward == -1) {
        this.reward = 0;
        this.lastTransitions.highPriority = [];
        this.lastTransitions.lowPriority = [];
      }

      this.agent.previousState = done ? null : currentState;
      this.agent.previousAction = done ? null : currentAction;

      this.agent.epsilon = Math.max(
        this.agent.finalEpsilon,
        this.agent.epsilon - 1 / this.agent.epsilonDecaySteps
      );
      return currentAction;
    };
    this.agent.argmax = nd => {
      return this.indexOfMax(nd.data);
    };
    this.agent.copy = target => {
      return ndarray(target.data.slice(), target.shape);
    };

    this.agent.getRandomSubarray = (arr, size) => {
      size = Math.min(size, arr.length);
      let shuffled = arr.slice(0),
        i = arr.length,
        min = i - size,
        temp,
        index;
      while (i-- > min) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
      }
      return shuffled.slice(min);
    };
    this.agent.learn = () => {
      if (
        this.agent.transitions.length < this.agent.learnBatchSize * 2 ||
        this.agent.transitionsLowPriority.length < this.agent.learnBatchSize
      ) {
        return;
      }
      let transitionsHighPriority = this.agent.getRandomSubarray(
        this.agent.transitions,
        this.agent.learnBatchSize
      );
      let transitionsLowPriority = this.agent.getRandomSubarray(
        this.agent.transitionsLowPriority,
        this.agent.learnBatchSize
      );
      // console.log("l", transitionsLowPriority);

      let transitions = [];
      for (let i = 0; i < this.agent.learnBatchSize; i++) {
        if (i < this.agent.theta * this.agent.learnBatchSize) {
          transitions.push(transitionsHighPriority[i]);
        } else {
          transitions.push(transitionsLowPriority[i]);
        }
      }
      console.log(transitionsLowPriority);

      this.agent.theta = Math.max(0.5, this.agent.theta * 0.98);
      let batchLoss = 0;
      transitions.forEach((t, k) => {
        // q(s, a) -> r + gamma * max_a' q(s', a')
        const qPrime = this.agent.copy(this.model.forward(t[3]));
        const q = this.model.forward(t[0]);
        const target = this.agent.copy(q);
        const reward = t[2];
        if (t[4]) {
          target.data[t[1]] = reward;
        } else {
          target.data[t[1]] =
            reward + this.agent.gamma * qPrime.data[this.agent.argmax(qPrime)];
        }

        const [loss, gradInputs] = this.model.criterion(q, target);
        this.agent.batchLoss += loss;
        if (this.agent.maxError) {
          gradInputs.data.forEach((v, k) => {
            if (Math.abs(v) > this.agent.maxError) {
              gradInputs.data[k] =
                v > 0 ? this.agent.maxError : -this.agent.maxError;
            }
          });
        }

        this.model.backward(gradInputs);
        this.model.update();
      });

      return transitions.length ? batchLoss / transitions.length : 0;
    };
  }

  indexOfMax(arr) {
    if (arr.length === 0) {
      return -1;
    }
    let max = -9999;
    let maxIndex = -1;
    let allPossibleMoves = this.getAllPossibleMoves();
    for (let i = 0; i < arr.length; i++) {
      let moveIsPossible = allPossibleMoves.indexOf(i) !== -1;
      if (arr[i] > max && moveIsPossible) {
        maxIndex = i;
        max = arr[i];
      }
    }

    return maxIndex;
  }

  stopLearning(): void {
    if (this.intervalSub != null) {
      this.intervalSub.unsubscribe();
    }
  }
  startLearning(period: number): void {
    this.highScore = 0;
    this.gameNo = 0;
    this.stopLearning();
    this.action = this.getAllPossibleMoves()[this.getRandomInt(0, 3)];
    this.makeMove();
    this.redraw();
    this.intervalSub = interval(period).subscribe(i => {
      if (this.action == null) {
        return;
      }
      if (this.done) {
        this.gameNo++;
        if (this.score > this.highScore) {
          this.highScore = this.score;
          console.log(this.highScore);
        }
        this.reset();
      }
      this.learn();
      i++;
    });
  }

  mapBoardToArray(): number[] {
    let array: number[] = [];
    for (let i = 0; i < this.HEIGHT / this.gridScale; i++) {
      for (let j = 0; j < this.WIDTH / this.gridScale; j++) {
        if (
          this.isMoveHittingSnake({ x: i, y: j }) ||
          i === -1 ||
          j === 1 ||
          i === this.HEIGHT / this.gridScale ||
          j === this.WIDTH / this.gridScale
        ) {
          array.push(-1);
        } else if (this.foodSquare.x === j && this.foodSquare.y === i) {
          array.push(1);
        } else {
          array.push(0);
        }
      }
    }
    return array;
  }

  getSimpleBoardRepresentation(): number[] {
    let result: number[] = [];

    result = result.concat(this.getLastACtionAsInput());

    let currentCoord: Coord = this.snake[this.snake.length - 1];
    let leftCoord: Coord = { x: currentCoord.x, y: currentCoord.y };
    leftCoord.x--;
    result.push(this.isMoveLosing(leftCoord) ? 1 : 0);

    let rightCoord: Coord = { x: currentCoord.x, y: currentCoord.y };
    rightCoord.x++;
    result.push(this.isMoveLosing(rightCoord) ? 1 : 0);

    let topCoord: Coord = { x: currentCoord.x, y: currentCoord.y };
    topCoord.y--;
    result.push(this.isMoveLosing(topCoord) ? 1 : 0);

    let botCoord: Coord = { x: currentCoord.x, y: currentCoord.y };
    botCoord.y++;
    result.push(this.isMoveLosing(botCoord) ? 1 : 0);

    result = result.concat(
      this.normalize(
        this.getAngleBetweenTwoPoints(currentCoord, this.foodSquare),
        -180,
        180
      )
    );
    return result;
  }

  getLastACtionAsInput(): number[] {
    let lastActionAsInput: number[];
    switch (this.action) {
      case Move.DOWN: {
        lastActionAsInput = [1, 0, 0, 0];
        break;
      }
      case Move.UP: {
        lastActionAsInput = [0, 1, 0, 0];

        break;
      }
      case Move.LEFT: {
        lastActionAsInput = [0, 0, 1, 0];

        break;
      }
      case Move.RIGHT: {
        lastActionAsInput = [0, 0, 0, 1];
        break;
      }
    }
    return lastActionAsInput;
  }

  getAngleBetweenTwoPoints(point1: Coord, point2: Coord): number {
    return (
      (Math.atan2(point2.y - point1.y, point2.x - point1.x) * 180) / Math.PI
    );
  }

  getVisionAsInput(): number[] {
    let currentCoord: Coord = this.snake[this.snake.length - 1];
    let allCoords: Coord[] = this.getAllCoords();
    let leftTopDiagonal: Coord[] = allCoords
      .filter(coord => {
        let isDiagonal: boolean =
          Math.abs(currentCoord.x - coord.x) ===
          Math.abs(currentCoord.y - coord.y);
        let isLeftTop: boolean =
          coord.x < currentCoord.x && coord.y > currentCoord.y;
        return isLeftTop && isDiagonal;
      })
      .sort(this.sortByDistanceComaprator(currentCoord));

    let rightTopDiagonal: Coord[] = allCoords
      .filter(coord => {
        let isDiagonal: boolean =
          Math.abs(currentCoord.x - coord.x) ===
          Math.abs(currentCoord.y - coord.y);
        let isRightTop: boolean =
          coord.x > currentCoord.x && coord.y > currentCoord.y;
        return isRightTop && isDiagonal;
      })
      .sort(this.sortByDistanceComaprator(currentCoord));

    let leftBotDiagonal: Coord[] = allCoords
      .filter(coord => {
        let isDiagonal: boolean =
          Math.abs(currentCoord.x - coord.x) ===
          Math.abs(currentCoord.y - coord.y);
        let isLeftBot: boolean =
          coord.x < currentCoord.x && coord.y < currentCoord.y;
        return isLeftBot && isDiagonal;
      })
      .sort(this.sortByDistanceComaprator(currentCoord));

    let rightBotDiagonal: Coord[] = allCoords
      .filter(coord => {
        let isDiagonal: boolean =
          Math.abs(currentCoord.x - coord.x) ===
          Math.abs(currentCoord.y - coord.y);
        let isRightBot: boolean =
          coord.x > currentCoord.x && coord.y < currentCoord.y;
        return isRightBot && isDiagonal;
      })
      .sort(this.sortByDistanceComaprator(currentCoord));

    let leftDirection: Coord[] = allCoords.filter(coord => {
      return coord.y == currentCoord.y && coord.x < currentCoord.x;
    });

    let rightDirection: Coord[] = allCoords.filter(coord => {
      return coord.y == currentCoord.y && coord.x > currentCoord.x;
    });

    let topDirection: Coord[] = allCoords.filter(coord => {
      return coord.x == currentCoord.x && coord.y > currentCoord.y;
    });

    let botDirection: Coord[] = allCoords.filter(coord => {
      return coord.x == currentCoord.x && coord.y < currentCoord.y;
    });

    let result: number[] = [];
    [
      this.getInputsFromDirection(leftTopDiagonal, true),
      this.getInputsFromDirection(rightTopDiagonal, true),
      this.getInputsFromDirection(leftBotDiagonal, true),
      this.getInputsFromDirection(rightBotDiagonal, true),
      this.getInputsFromDirection(rightDirection, false),
      this.getInputsFromDirection(leftDirection, false),
      this.getInputsFromDirection(topDirection, false),
      this.getInputsFromDirection(botDirection, false)
    ].forEach(info => {
      result.push(info.distanceToFood);
      result.push(info.distanceToSnake);
      result.push(info.distanceToWall);
    });

    result = result.concat(this.getLastACtionAsInput());
    result = result.concat(this.getTailDirection());

    return result;
  }

  getTailDirection(): number[] {
    let lastTailCoord: Coord = this.snake[0];
    let previous: Coord = this.snake[1];
    let result: number[];
    if (lastTailCoord.x < previous.x) {
      result = [0, 0, 0, 1];
    }
    if (lastTailCoord.y < previous.y) {
      result = [0, 0, 1, 0];
    }
    if (lastTailCoord.x > previous.x) {
      result = [0, 1, 0, 0];
    }
    if (lastTailCoord.y > previous.y) {
      result = [1, 0, 0, 0];
    }
    return result;
  }

  printDiagonals(
    leftTopDiagonal,
    rightTopDiagonal,
    leftBotDiagonal,
    topDirection,
    botDirection,
    rightBotDiagonal,
    leftDirection,
    rightDirection
  ): void {
    setTimeout(() => {
      this.ctx.fillStyle = "gray";
      leftTopDiagonal.forEach(snakeCoord => {
        this.ctx.fillRect(
          snakeCoord.x * this.gridScale,
          snakeCoord.y * this.gridScale,
          this.gridScale - 2,
          this.gridScale - 2
        );
      });
      rightTopDiagonal.forEach(snakeCoord => {
        this.ctx.fillRect(
          snakeCoord.x * this.gridScale,
          snakeCoord.y * this.gridScale,
          this.gridScale - 2,
          this.gridScale - 2
        );
      });
      leftBotDiagonal.forEach(snakeCoord => {
        this.ctx.fillRect(
          snakeCoord.x * this.gridScale,
          snakeCoord.y * this.gridScale,
          this.gridScale - 2,
          this.gridScale - 2
        );
      });
      topDirection.forEach(snakeCoord => {
        this.ctx.fillRect(
          snakeCoord.x * this.gridScale,
          snakeCoord.y * this.gridScale,
          this.gridScale - 2,
          this.gridScale - 2
        );
      });
      botDirection.forEach(snakeCoord => {
        this.ctx.fillRect(
          snakeCoord.x * this.gridScale,
          snakeCoord.y * this.gridScale,
          this.gridScale - 2,
          this.gridScale - 2
        );
      });
      rightBotDiagonal.forEach(snakeCoord => {
        this.ctx.fillRect(
          snakeCoord.x * this.gridScale,
          snakeCoord.y * this.gridScale,
          this.gridScale - 2,
          this.gridScale - 2
        );
      });
      leftDirection.forEach(snakeCoord => {
        this.ctx.fillRect(
          snakeCoord.x * this.gridScale,
          snakeCoord.y * this.gridScale,
          this.gridScale - 2,
          this.gridScale - 2
        );
      });
      rightDirection.forEach(snakeCoord => {
        this.ctx.fillRect(
          snakeCoord.x * this.gridScale,
          snakeCoord.y * this.gridScale,
          this.gridScale - 2,
          this.gridScale - 2
        );
      });
    });
  }

  getAllCoords(): Coord[] {
    let allCoords: Coord[] = [];
    for (var i = 0; i < this.HEIGHT / this.gridScale; i++) {
      for (var j = 0; j < this.WIDTH / this.gridScale; j++) {
        allCoords.push({ x: j, y: i });
      }
    }
    return allCoords;
  }

  getInputsFromDirection(
    arrayOfDirection: Coord[],
    isDiagonal: boolean
  ): DirectionInfo {
    const INFI: number = 9999;
    let distanceToFood: number = INFI;
    let distanceToSnake: number = INFI;
    let distanceToWall: number = arrayOfDirection.length;
    for (let i = 0; i < arrayOfDirection.length; i++) {
      let candidate: Coord = arrayOfDirection[i];
      if (
        candidate.x == this.foodSquare.x &&
        this.foodSquare.y == candidate.y
      ) {
        distanceToFood = i + 1;
      }
      if (
        this.snake.some(
          coord => candidate.x == coord.x && coord.y == candidate.y
        )
      ) {
        distanceToSnake = i + 1;
      }
    }
    let maxDiagonalDistance: number = Math.floor(
      this.getDistance(
        0,
        0,
        this.WIDTH / this.gridScale,
        this.HEIGHT / this.gridScale
      )
    );

    let maxVerticalDistance: number = this.WIDTH / this.gridScale;

    return {
      distanceToFood:
        distanceToFood == INFI
          ? 1
          : this.normalize(
              distanceToFood,
              0,
              isDiagonal ? maxDiagonalDistance : maxVerticalDistance
            ),
      distanceToSnake:
        distanceToSnake == INFI
          ? 1
          : this.normalize(
              distanceToSnake,
              0,
              isDiagonal ? maxDiagonalDistance : maxVerticalDistance
            ),
      distanceToWall: this.normalize(
        distanceToWall,
        0,
        isDiagonal ? maxDiagonalDistance : maxVerticalDistance
      )
    };
  }

  normalize(toBeNormalized: number, min: number, max: number): number {
    return (toBeNormalized - min) / (max - min);
  }

  sortByDistanceComaprator(current: Coord): (c1, c2) => number {
    return (c1, c2) =>
      this.getDistance(c1.x, current.x, c1.y, current.y) -
      this.getDistance(c2.x, current.x, c2.y, current.y);
  }

  reset(): void {
    this.done = false;
    this.score = 0;
    this.reward = 0;
    this.foodSquare = null;
    this.action = Move.DOWN;
    this.initSnake();
    this.dropFoodOnAvailableSquare();
    // if (this.intervalSub != null){
    //   this.intervalSub.unsubscribe();
    // }
    // this.intervalSub = interval(100).subscribe(i => {
    //   this.makeMove();
    //   this.redraw();
    // });
  }

  learn(): void {
    let observation = ndarray(this.getVisionAsInput());
    let reward = this.reward;
    let done = this.done;

    let action = this.agent.step(observation, reward, done);

    let maxIterations: number = 0.7 * this.snake.length + 10;
    if (this.movesSinceLastEating >= maxIterations) {
      this.reward = -0.5 / this.snake.length;
      this.movesSinceLastEating = 0;
      this.lastTransitions.highPriority.forEach(index => {
        this.agent.transitions[index][2] = this.reward;
      });
      this.lastTransitions.highPriority = [];
      this.lastTransitions.lowPriority.forEach(index => {
        this.agent.transitionsLowPriority[index][2] = this.reward;
      });
      this.lastTransitions.lowPriority = [];
    //   let i = 0;
    //   while (i < maxIterations) {
    //     this.agent.transitions[
    //       (this.agent.transitionCount - 1 - i) % this.agent.memorySize
    //     ][2] = this.reward;
    //     i++;
    //   }
    }

    // if (this.gameNo > 32) {
    let loss = this.agent.learn();
    // console.log(' loss: ', loss);
    // }

    let predictedAction: Move = Move[Move[action]];

    this.action = predictedAction;
    this.makeMove();
    // if (this.gameNo > 200){
    setTimeout(() => {
      this.redraw();
    });
    // this.redraw();
    // let maxIterations: number = 0.7 * this.snake.length + 10;

    // }
  }

  redraw(): void {
    this.ctx.fillStyle = "black";
    this.ctx.fillRect(0, 0, this.WIDTH, this.HEIGHT);

    this.ctx.fillStyle = "lime";
    this.ctx.fillRect(
      this.foodSquare.x * this.gridScale,
      this.foodSquare.y * this.gridScale,
      this.gridScale - 2,
      this.gridScale - 2
    );
    this.ctx.fillStyle = "red";
    this.snake.forEach(snakeCoord => {
      this.ctx.fillRect(
        snakeCoord.x * this.gridScale,
        snakeCoord.y * this.gridScale,
        this.gridScale - 2,
        this.gridScale - 2
      );
    });
  }

  initSnake(): void {
    this.snake = [];
    this.snake.push({ x: 5, y: 5 }, { x: 6, y: 5 });
  }

  dropFoodOnAvailableSquare(): void {
    let array: Coord[] = [];
    for (let i = 0; i < this.HEIGHT / this.gridScale; i++) {
      for (let j = 0; j < this.WIDTH / this.gridScale; j++) {
        if (
          this.isMoveHittingSnake({ x: i, y: j }) ||
          i === -1 ||
          j === -1 ||
          i === this.HEIGHT / this.gridScale ||
          j === this.WIDTH / this.gridScale
        ) {
        } else if (
          this.foodSquare !== null &&
          this.foodSquare.x === j &&
          this.foodSquare.y === i
        ) {
        } else {
          array.push({ x: i, y: j });
        }
      }
    }
    let randomAvailable: Coord = array[this.getRandomInt(0, array.length - 1)];
    this.foodSquare = {
      x: randomAvailable.x,
      y: randomAvailable.y
    };
  }

  makeMove(): void {
    let lastSnakeCoord = this.snake[this.snake.length - 1];
    let currentCoord = { x: lastSnakeCoord.x, y: lastSnakeCoord.y };
    switch (this.action) {
      case Move.LEFT: {
        currentCoord.x = currentCoord.x - 1;
        break;
      }
      case Move.RIGHT: {
        currentCoord.x = currentCoord.x + 1;
        break;
      }
      case Move.UP: {
        currentCoord.y = currentCoord.y - 1;
        break;
      }
      case Move.DOWN: {
        currentCoord.y = currentCoord.y + 1;
        break;
      }
    }

    if (this.isMoveLosing(currentCoord)) {
      // alert("koniec gry");
      this.done = true;
      this.reward = -1.0;
      this.movesSinceLastEating = 0;
      return;
    }
    this.snake.push(currentCoord);

    if (
      currentCoord.x === this.foodSquare.x &&
      currentCoord.y === this.foodSquare.y
    ) {
      this.dropFoodOnAvailableSquare();
      this.score += 1;
      this.reward = 1.0;
      this.movesSinceLastEating = 0;
    } else {
      this.snake.shift();
      let newCurrentDistanceToFood: number = this.getDistance(
        currentCoord.x,
        currentCoord.y,
        this.foodSquare.x,
        this.foodSquare.y
      );
      // this.reward =
      //   newCurrentDistanceToFood < this.currentDistanceToFood
      //     ? 0.1
      //     : newCurrentDistanceToFood == this.currentDistanceToFood
      //     ? 0
      //     : (this.reward = -0.2);

      this.movesSinceLastEating++;

      this.reward = Math.max(
        -1,
        Math.min(
          1,
          this.reward +
            this.calculateRewardByDistance(lastSnakeCoord, currentCoord)
        )
      );
      this.currentDistanceToFood = newCurrentDistanceToFood;
    }
  }

  calculateRewardByDistance(curr: Coord, next: Coord): number {
    let currDistance: number = this.getDistance(
      curr.x,
      curr.y,
      this.foodSquare.x,
      this.foodSquare.y
    );
    let nextDistance: number = this.getDistance(
      next.x,
      next.y,
      this.foodSquare.x,
      this.foodSquare.y
    );
    let result: number =
      Math.log(
        (this.snake.length + currDistance) / (this.snake.length + nextDistance)
      ) / Math.log(this.snake.length);
    if (result < -1) {
      result = -1;
    }
    if (result > 1) {
      result = 1;
    }
    return result;
  }

  getTrainingGap(): number {
    const k: number = 10;
    const p: number = 0.4;
    const q: number = 2;

    if (this.snake.length <= k) {
      return 0.5 * (this.WIDTH / this.gridScale);
    }
    return p * this.snake.length + 2;
  }

  isMoveLosing(candidateCoord: Coord): boolean {
    let isCandidateCoordHittingSnake: boolean = this.snake.some(
      coord => coord.x === candidateCoord.x && coord.y === candidateCoord.y
    );
    return (
      candidateCoord.x === this.WIDTH / this.gridScale ||
      candidateCoord.y === this.HEIGHT / this.gridScale ||
      candidateCoord.x === -1 ||
      candidateCoord.y === -1 ||
      isCandidateCoordHittingSnake
    );
  }

  getRandomInt(min, max): number {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min)) + min;
  }

  isMoveHittingSnake(move: Coord): boolean {
    return this.snake.some(coord => coord.x === move.x && coord.y === move.y);
  }

  getAllPossibleMoves(): Move[] {
    let allPossible: Move[];
    switch (this.action) {
      case Move.DOWN: {
        allPossible = [Move.LEFT, Move.RIGHT, Move.DOWN];
        break;
      }
      case Move.UP: {
        allPossible = [Move.LEFT, Move.RIGHT, Move.UP];

        break;
      }
      case Move.LEFT: {
        allPossible = [Move.DOWN, Move.LEFT, Move.UP];

        break;
      }
      case Move.RIGHT: {
        allPossible = [Move.DOWN, Move.UP, Move.RIGHT];
        break;
      }
      // default:{
      //   allPossible = [Move.UP];
      //   break;
      // }
    }
    return allPossible;
  }

  getDistance(xA, yA, xB, yB): number {
    let xDiff = xA - xB;
    let yDiff = yA - yB;

    return Math.sqrt(xDiff * xDiff + yDiff * yDiff);
  }
}

export class Coord {
  public x: number;
  public y: number;
}

export enum Move {
  UP = 0,
  LEFT = 1,
  RIGHT = 2,
  DOWN = 3
}

export class DirectionInfo {
  public distanceToWall: number;
  public distanceToFood: number;
  public distanceToSnake: number;
}

export interface LastTransitionsIndexes {
  highPriority: number[];
  lowPriority: number[];
}
