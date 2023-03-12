import Keyv from 'keyv'
import pTimeout from 'p-timeout'
import QuickLRU from 'quick-lru'
import { v4 as uuidv4 } from 'uuid'

import * as tokenizer from './tokenizer'
import * as types from './types'
import { ChatGPTAPI } from './chatgpt-api'
import { fetch as globalFetch } from './fetch'
import { fetchSSE } from './fetch-sse'

const CHATGPT_MODEL = 'gpt-3.5-turbo'

const USER_LABEL_DEFAULT = 'User'
const ASSISTANT_LABEL_DEFAULT = 'ChatGPT'

interface TokenPooling {
  addToken(token: string): TokenInfo
  getCurrentToken(): TokenInfo | undefined
  getToken(id: number): TokenInfo | undefined
  updateToken(id: number, updates: Partial<TokenInfo>): TokenInfo | undefined
  deleteToken(id: number): boolean
  listTokens(): TokenInfo[]
}

export type TokenInfo = {
  id: number
  token: string
  usage?: number
  limit?: number
}

type TokenSelectionAlgorithm = (
  tokens: TokenInfo[],
  lastSelectedIndex: number
) => TokenInfo | undefined

export class ChatGPTPool extends ChatGPTAPI implements TokenPooling {
  private tokens: TokenInfo[]
  private selectionAlgorithm: TokenSelectionAlgorithm
  currentToken: TokenInfo

  /**
   * Creates a new client wrapper around OpenAI's chat completion API, mimicing the official ChatGPT webapp's functionality as closely as possible.
   *
   * @param apiKey - OpenAI API key (required).
   * @param apiBaseUrl - Optional override for the OpenAI API base URL.
   * @param debug - Optional enables logging debugging info to stdout.
   * @param completionParams - Param overrides to send to the [OpenAI chat completion API](https://platform.openai.com/docs/api-reference/chat/create). Options like `temperature` and `presence_penalty` can be tweaked to change the personality of the assistant.
   * @param maxModelTokens - Optional override for the maximum number of tokens allowed by the model's context. Defaults to 4096.
   * @param maxResponseTokens - Optional override for the minimum number of tokens allowed for the model's response. Defaults to 1000.
   * @param messageStore - Optional [Keyv](https://github.com/jaredwray/keyv) store to persist chat messages to. If not provided, messages will be lost when the process exits.
   * @param getMessageById - Optional function to retrieve a message by its ID. If not provided, the default implementation will be used (using an in-memory `messageStore`).
   * @param upsertMessage - Optional function to insert or update a message. If not provided, the default implementation will be used (using an in-memory `messageStore`).
   * @param fetch - Optional override for the `fetch` implementation to use. Defaults to the global `fetch` function.
   */
  constructor(opts: types.ChatGPTAPIOptions) {
    super(opts)
    const { apiKey } = opts

    this.tokens = []
    this.addToken(apiKey)
    this.selectionAlgorithm = this.defaultSelectionAlgorithm
  }

  /**
   * Sends a message to the OpenAI chat completions endpoint, waits for the response
   * to resolve, and returns the response.
   *
   * If you want your response to have historical context, you must provide a valid `parentMessageId`.
   *
   * If you want to receive a stream of partial responses, use `opts.onProgress`.
   *
   * Set `debug: true` in the `ChatGPTAPI` constructor to log more info on the full prompt sent to the OpenAI chat completions API. You can override the `systemMessage` in `opts` to customize the assistant's instructions.
   *
   * @param message - The prompt message to send
   * @param opts.parentMessageId - Optional ID of the previous message in the conversation (defaults to `undefined`)
   * @param opts.messageId - Optional ID of the message to send (defaults to a random UUID)
   * @param opts.systemMessage - Optional override for the chat "system message" which acts as instructions to the model (defaults to the ChatGPT system message)
   * @param opts.timeoutMs - Optional timeout in milliseconds (defaults to no timeout)
   * @param opts.onProgress - Optional callback which will be invoked every time the partial response is updated
   * @param opts.abortSignal - Optional callback used to abort the underlying `fetch` call using an [AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController)
   *
   * @returns The response from ChatGPT
   */
  async sendMessage(
    text: string,
    opts: types.SendMessageOptions = {}
  ): Promise<types.ChatMessage> {
    this.currentToken = this.selectToken()
    if (!this.currentToken) {
      throw new Error(
        'OpenAI missing required apiKey in token pool. Please contanct administrator. | 无法从API密钥池获取Token，请联系管理员'
      )
    }
    this._apiKey = this.currentToken.token

    // add pre-check and post-process hooks
    const { preCheckHooks = [], postProcessHooks = [] } = opts

    preCheckHooks.push(this.preCheck.bind(this))
    postProcessHooks.push(this.postProcess.bind(this))

    return super.sendMessage(text, opts)
  }

  preCheck(numToken: number): boolean {
    this.currentToken = this.updateToken(this.currentToken.id, {
      usage: this.currentToken.usage ?? 0 + numToken
    })
    return true
  }

  postProcess(numToken: number) {
    this.currentToken = this.updateToken(this.currentToken.id, {
      usage: this.currentToken.usage ?? 0 + numToken
    })
  }

  // Implement the TokenPooling interface methods
  addToken(token: string): TokenInfo {
    const id = this.tokens.length + 1
    const newToken: TokenInfo = { id, token }
    this.tokens.push(newToken)
    return newToken
  }

  getCurrentToken(): TokenInfo | undefined {
    return this.currentToken
  }

  getToken(id: number): TokenInfo | undefined {
    return this.tokens.find((token) => token.id === id)
  }

  updateToken(id: number, updates: Partial<TokenInfo>): TokenInfo | undefined {
    const index = this.tokens.findIndex((token) => token.id === id)
    if (index !== -1) {
      this.tokens[index] = { ...this.tokens[index], ...updates }
      return this.tokens[index]
    }
    return undefined
  }

  deleteToken(id: number): boolean {
    const index = this.tokens.findIndex((token) => token.id === id)
    if (index !== -1) {
      this.tokens.splice(index, 1)
      return true
    }
    return false
  }

  listTokens(): TokenInfo[] {
    return this.tokens
  }

  selectToken(): TokenInfo | undefined {
    return this.selectionAlgorithm(this.tokens, this.lastSelectedIndex)
  }

  setSelectionAlgorithm(algorithm: TokenSelectionAlgorithm): void {
    this.selectionAlgorithm = algorithm
  }

  private lastSelectedIndex: number = 0

  private defaultSelectionAlgorithm(
    tokens: TokenInfo[],
    lastSelectedIndex: number
  ): TokenInfo | undefined {
    const availableTokens = tokens.filter(
      (token, index) =>
        lastSelectedIndex ||
        index > lastSelectedIndex ||
        !token.limit ||
        !token.usage ||
        token.usage < token.limit
    )

    if (availableTokens.length === 0) {
      console.log(
        `no availabe token found in token pool of size ${tokens.length}`
      )
      return undefined
    }

    const token =
      availableTokens[this.lastSelectedIndex++ % availableTokens.length]

    return token
  }
}
