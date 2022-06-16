import config from "./config"
const axios = require("axios") 


class FastAPIClient {
	constructor(overrides) {
		this.config = {
			...config,
			...overrides,
		}

		this.apiClient = this.getApiClient(this.config) 
	}

	/* Create Axios client instance pointing at the REST api backend */
	getApiClient(config) {
		let initialConfig = {
			baseURL: `${config.apiBasePath}`,
		}
		let client = axios.create(initialConfig)
		return client
	}

	getPrediction(msg) {
		return this.apiClient.get(`/predict?msg=${msg}`).then(({data}) => {  
			return data
		})
	}

	getAllStats() {
		return this.apiClient.get(`/allstats`).then(({data}) => {  
			return data
		})
	}

	getTypeStats(type) {
		return this.apiClient.get(`/typestats?type=${type}`).then(({data}) => {  
			return data
		})
	}

	updateDataset(new_type,msg) {
		return this.apiClient.post(`/updatemodel?type=${new_type}&msg=${msg}`).then(() => {})
	}
}
export default FastAPIClient;